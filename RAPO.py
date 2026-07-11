import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datasets import load_dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
import pandas as pd
from datasetload import AnswerGradingDataset,align_binary_labels,compute_class_weights,load_raw_dataset,label_distribution
from metrics import compute_metrics

def compute_ece(probs, labels, n_bins=10):

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)

    accuracies = predictions == labels

    bins = np.linspace(0,1,n_bins+1)

    ece = 0

    for i in range(n_bins):

        mask = (confidences > bins[i]) & (confidences <= bins[i+1])

        if np.sum(mask)==0:
            continue

        acc = accuracies[mask].mean()

        conf = confidences[mask].mean()

        ece += np.abs(acc-conf) * np.sum(mask)/len(labels)

    return ece

def parse_args():
    parser = argparse.ArgumentParser()

    # 模型与数据
    parser.add_argument("--model_name", type=str, default="/home/shared/winston/model/microsoft/deberta-v3-base")
    parser.add_argument("--dataset", type=str, default="scientsbank", choices=["scientsbank", "beetle"])
    parser.add_argument("--output_dir", type=str, default="results_ce_ppo")
    parser.add_argument("--selection_split", type=str, default="test_ua",
                        help="用于选择最优 checkpoint 的 split。默认沿用 baseline 的 test_ua。")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # RL / PPO 阶段
    parser.add_argument("--rl_epochs", type=int, default=2)
    parser.add_argument("--rl_lr", type=float, default=1.0e-5)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--ppo_clip", type=float, default=0.2
                        )

    # 多目标损失
    parser.add_argument("--ce_coef", type=float, default=0.6,
                        help="RL 阶段 CE 辅助损失系数。")
    parser.add_argument("--rl_coef", type=float, default=0.4,
                        help="RL 阶段 PPO policy loss 系数。")
    parser.add_argument("--kl_coef", type=float, default=0.02,
                        help="固定 SFT 参考模型 KL 正则系数；设为 0 可节省显存。")
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help="策略熵奖励系数，帮助探索。")

    # 奖励设计
    parser.add_argument("--correct_reward", type=float, default=1.0)
    parser.add_argument("--wrong_reward", type=float, default=-1.0)
    parser.add_argument("--false_negative_reward", type=float, default=-1.5,
                        help="真实 correct 却预测 incorrect 的奖励。")
    parser.add_argument("--false_positive_reward", type=float, default=-1.2,
                        help="真实 incorrect 却预测 correct 的奖励。")

    # 训练稳定性
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--disable_class_weights", action="store_true",
                        help="关闭按类别频率计算的 class weight。")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--force_fp32", action="store_true",
                        help="强制使用 fp32；否则 CUDA 上优先尝试 bfloat16。")

    return parser.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def choose_dtype(force_fp32: bool) -> torch.dtype:
    if force_fp32:
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def safe_float(x: torch.Tensor) -> torch.Tensor:
    return x.float() if x.dtype != torch.float32 else x


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)





def build_loader(hf_dataset, tokenizer, batch_size: int, max_length: int,
                 shuffle: bool, num_workers: int) -> DataLoader:
    ds = AnswerGradingDataset(hf_dataset, tokenizer, max_length=max_length)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

@torch.no_grad()
def evaluate(model, hf_dataset, tokenizer, args, device: torch.device) -> Dict[str, float]:
    loader = build_loader(
        hf_dataset=hf_dataset,
        tokenizer=tokenizer,
        batch_size=args.eval_batch_size,
        max_length=args.max_length,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model.eval()
    preds_list, labels_list = [], []
    logits_list = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        outputs = model(**batch)
        logits = safe_float(outputs.logits)
        prob = torch.softmax(logits, dim=-1)
        logits_list.extend(prob.detach().cpu().tolist())
        # confidence, prediction = prob.max(dim=1)
        preds = torch.argmax(logits, dim=-1)

        preds_list.extend(preds.detach().cpu().tolist())
        labels_list.extend(labels.detach().cpu().tolist())

    ece = compute_ece(logits_list,labels_list)
    print('ECE:',ece)

    return compute_metrics(preds_list, labels_list)


def evaluate_all_splits(model, test_data, tokenizer, args, device: torch.device) -> Dict[str, Dict[str, float]]:
    results = {}
    for split_name in test_data.keys():
        print(f"[Eval] split={split_name}")
        results[split_name] = evaluate(model, test_data[split_name], tokenizer, args, device)
    return results


def save_metrics(metrics_by_split: Dict[str, Dict[str, float]], save_prefix: Path):
    """
    统一保存 JSON / CSV / XLSX（若 pandas + excel 引擎可用）。
    """
    ensure_dir(save_prefix.parent)

    json_path = save_prefix.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_by_split, f, ensure_ascii=False, indent=2)

    rows = []
    for split_name, metrics in metrics_by_split.items():
        row = {"split": split_name}
        row.update(metrics)
        rows.append(row)

    # 无 pandas 时，至少 JSON 已保存
    if pd is None:
        return

    df = pd.DataFrame(rows)
    csv_path = save_prefix.with_suffix(".csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    xlsx_path = save_prefix.with_suffix(".xlsx")
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception as e:
        print(f"[Warn] 保存 xlsx 失败，仅保留 json/csv：{e}")


# =========================
# 模型与优化器
# =========================

def build_model(model_name_or_dir: str, model_dtype: torch.dtype, device: torch.device,
                gradient_checkpointing: bool):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_dir,
        num_labels=2,
        dtype=torch.float32
        # trust_remote_code=True,
        # use_cache=False,
    )
    model.to(device)
    return model


def create_optimizer_scheduler(model, lr: float, weight_decay: float,
                               total_steps: int, warmup_ratio: float):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_steps, 1),
    )
    return optimizer, scheduler


def save_checkpoint(model, tokenizer, save_dir: Path):
    ensure_dir(save_dir)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def print_metrics_brief(prefix: str, metrics: Dict[str, float]):
    print(
        f"{prefix} | "
        f"accuracy={metrics.get('accuracy', 0.0):.4f}, "
        f"macro_f1={metrics.get('macro_f1', 0.0):.4f}, "
        f"kappa={metrics.get('cohen_kappa', 0.0):.4f}"
    )



def compute_rewards(
    actions: torch.Tensor,
    labels: torch.Tensor,
    args,
) -> torch.Tensor:
    rewards = torch.full_like(labels, fill_value=args.wrong_reward, dtype=torch.float32)

    correct_mask = actions.eq(labels)
    rewards[correct_mask] = args.correct_reward

    false_negative_mask = labels.eq(1) & actions.eq(0)
    false_positive_mask = labels.eq(0) & actions.eq(1)

    rewards[false_negative_mask] = args.false_negative_reward
    rewards[false_positive_mask] = args.false_positive_reward

    return rewards


def normalize_advantages(rewards: torch.Tensor) -> torch.Tensor:
    advantages = rewards - rewards.mean()
    std = rewards.std(unbiased=False)
    if torch.isfinite(std) and std.item() > 1e-6:
        advantages = advantages / (std + 1e-8)
    return advantages


def train_ppo_style_rl(
    model,
    ref_model,
    train_data,
    test_data,
    tokenizer,
    args,
    device: torch.device,
    class_weights: Optional[torch.Tensor],
    run_dir: Path,
) -> Tuple[Path, Dict[str, Dict[str, float]]]:

    train_loader = build_loader(
        hf_dataset=train_data,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True,
        num_workers=args.num_workers,
    )

    total_steps = len(train_loader) * args.rl_epochs * args.ppo_epochs
    optimizer, scheduler = create_optimizer_scheduler(
        model=model,
        lr=args.rl_lr,
        weight_decay=args.weight_decay,
        total_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
    )

    best_score = -1.0
    best_metrics_all = None
    best_dir = run_dir / "rl_best"

    if args.selection_split not in test_data:
        raise ValueError(
            f"selection_split={args.selection_split} 不存在，当前可用：{list(test_data.keys())}"
        )
    for epoch in range(1, args.rl_epochs + 1):
        model.train()

        epoch_total_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_policy_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_entropy = 0.0
        epoch_reward = 0.0
        batch_count = 0

        progress = tqdm(train_loader, desc=f"[RL] Epoch {epoch}/{args.rl_epochs}")

        for batch in progress:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # 1) 冻结当前策略的旧分布，并据此采样动作
            model.eval()
            with torch.no_grad():
                old_outputs = model(**batch)
                old_logits = safe_float(old_outputs.logits)
                old_dist = Categorical(logits=old_logits)
                actions = old_dist.sample()
                # actions = torch.argmax(old_logits, dim=-1)
                old_log_probs = old_dist.log_prob(actions).detach()

                if ref_model is not None:
                    ref_outputs = ref_model(**batch)
                    ref_logits = safe_float(ref_outputs.logits).detach()
                else:
                    ref_logits = None

            rewards = compute_rewards(actions, labels, args)
            advantages = normalize_advantages(rewards).detach()

            model.train()
            last_total_loss = None
            last_ce_loss = None
            last_policy_loss = None
            last_kl_loss = None
            last_entropy = None

            for _ in range(args.ppo_epochs):
                optimizer.zero_grad(set_to_none=True)

                outputs = model(**batch)

                logits = safe_float(outputs.logits)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions)

                ratio = torch.exp(new_log_probs - old_log_probs)
                surrogate_1 = ratio * advantages
                surrogate_2 = torch.clamp(
                    ratio,
                    1.0 - args.ppo_clip,
                    1.0 + args.ppo_clip,
                ) * advantages
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
                # ce_loss = F.cross_entropy(logits, labels)

                ce_loss = F.cross_entropy(logits, labels, weight=class_weights)
                entropy = dist.entropy().mean()

                if ref_logits is not None and args.kl_coef > 0.0:
                    kl_loss = F.kl_div(
                        F.log_softmax(logits, dim=-1),
                        F.softmax(ref_logits, dim=-1),
                        reduction="batchmean",
                    )
                else:
                    kl_loss = torch.zeros((), device=device, dtype=logits.dtype)

                total_loss = (
                    args.ce_coef * ce_loss
                    + args.rl_coef * policy_loss
                    + args.kl_coef * kl_loss
                    - args.entropy_coef * entropy
                )

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()

                last_total_loss = float(total_loss.item())
                last_ce_loss = float(ce_loss.item())
                last_policy_loss = float(policy_loss.item())
                last_kl_loss = float(kl_loss.item())
                last_entropy = float(entropy.item())

            batch_count += 1
            epoch_total_loss += last_total_loss if last_total_loss is not None else 0.0
            epoch_ce_loss += last_ce_loss if last_ce_loss is not None else 0.0
            epoch_policy_loss += last_policy_loss if last_policy_loss is not None else 0.0
            epoch_kl_loss += last_kl_loss if last_kl_loss is not None else 0.0
            epoch_entropy += last_entropy if last_entropy is not None else 0.0
            epoch_reward += float(rewards.mean().item())

            progress.set_postfix(
                total=f"{(last_total_loss or 0.0):.4f}",
                reward=f"{rewards.mean().item():.3f}",
            )

        denom = max(batch_count, 1)
        print(
            "[RL] "
            f"epoch={epoch}, "
            f"loss={epoch_total_loss/denom:.6f}, "
            f"ce={epoch_ce_loss/denom:.6f}, "
            f"policy={epoch_policy_loss/denom:.6f}, "
            f"kl={epoch_kl_loss/denom:.6f}, "
            f"entropy={epoch_entropy/denom:.6f}, "
            f"reward={epoch_reward/denom:.6f}"
        )

        selection_metrics = evaluate(
            model=model,
            hf_dataset=test_data[args.selection_split],
            tokenizer=tokenizer,
            args=args,
            device=device,
        )
        print_metrics_brief(f"[RL] {args.selection_split}", selection_metrics)

        score = selection_metrics["macro_f1"]
        if score > best_score:
            best_score = score
            print(f"[RL] 保存新的最优 checkpoint，macro_f1={best_score:.6f}")
            save_checkpoint(model, tokenizer, best_dir)

            best_metrics_all = evaluate_all_splits(
                model=model,
                test_data=test_data,
                tokenizer=tokenizer,
                args=args,
                device=device,
            )
            save_metrics(best_metrics_all, run_dir / "rl_best_metrics")

    if best_metrics_all is None:
        save_checkpoint(model, tokenizer, best_dir)
        best_metrics_all = evaluate_all_splits(model, test_data, tokenizer, args, device)
        save_metrics(best_metrics_all, run_dir / "rl_best_metrics")

    return best_dir, best_metrics_all



def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = choose_dtype(args.force_fp32)

    model_name_short = Path(args.model_name.rstrip("/")).name
    run_dir = Path(args.output_dir) / f"{model_name_short}_{args.dataset}"
    ensure_dir(run_dir)

    print("=" * 80)
    print(f"device={device}")
    print(f"dtype={model_dtype}")
    print(f"dataset={args.dataset}")
    print(f"output_dir={run_dir}")
    print("=" * 80)

    # 1) 数据加载
    train_data, test_data = load_raw_dataset(args.dataset)
    train_data = align_binary_labels(train_data)
    test_data = align_binary_labels(test_data)

    print("[Data] train label distribution:", label_distribution(train_data))
    for split_name in test_data.keys():
        print(f"[Data] {split_name} label distribution:", label_distribution(test_data[split_name]))

    # 2) tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3) 模型
    model = build_model(
        model_name_or_dir=args.model_name,
        model_dtype=model_dtype,
        device=device,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # 4) 类别权重
    class_weights = compute_class_weights(
        train_data=train_data,
        device=device,
        disable_class_weights=args.disable_class_weights,
    )
    if class_weights is None:
        print("[Train] class weights disabled.")
    else:
        print(f"[Train] class weights={class_weights.detach().cpu().tolist()}")

    sft_best_dir = '/home/shared/CL/PPO/results_ce_ppo/deberta-v3-base_scientsbank/sft_best'

    model = build_model(
        model_name_or_dir=str(sft_best_dir),
        model_dtype=model_dtype,
        device=device,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # 6) 固定参考模型（可选）
    if args.kl_coef > 0.0:
        print(f"[RL] kl_coef={args.kl_coef} > 0，加载固定 SFT reference model。")
        ref_model = build_model(
            model_name_or_dir=str(sft_best_dir),
            model_dtype=model_dtype,
            device=device,
            gradient_checkpointing=False,
        )
        ref_model.config.pad_token_id = tokenizer.pad_token_id
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
    else:
        print("[RL] kl_coef=0，跳过 reference model，可节省显存。")
        ref_model = None

   
    print("\n" + "=" * 80)
    print("[Stage 2] PPO-style RL fine-tuning")
    print("=" * 80)
    rl_best_dir, rl_metrics = train_ppo_style_rl(
        model=model,
        ref_model=ref_model,
        train_data=train_data,
        test_data=test_data,
        tokenizer=tokenizer,
        args=args,
        device=device,
        class_weights=class_weights,
        run_dir=run_dir,
    )

    print("[Stage 2] best checkpoint:", rl_best_dir)
    for split_name, metrics in rl_metrics.items():
        print_metrics_brief(f"[Stage 2][Best] {split_name}", metrics)


if __name__ == "__main__":
    main()
