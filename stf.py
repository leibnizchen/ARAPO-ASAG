
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from datasetload import AnswerGradingDataset,align_binary_labels,compute_class_weights,load_raw_dataset,label_distribution
from metrics import compute_metrics
import pandas as pd

import torch
import torch.nn.functional as F
from uitls import ensure_dir,save_metrics


def FocalLoss(logits, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)   # pt = predicted probability of the true class
    loss = alpha * (1 - pt) ** gamma * ce_loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def parse_args():
    parser = argparse.ArgumentParser()
    # 模型与数据
    parser.add_argument("--model_name", type=str, default="/home/shared/winston/model/microsoft/deberta-v3-base")
    parser.add_argument("--dataset", type=str, default="beetle", choices=["scientsbank", "beetle"])
    parser.add_argument("--output_dir", type=str, default="results_ce_ppo")
    parser.add_argument("--selection_split", type=str, default="test_ua")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--sft_epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--disable_class_weights", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--force_fp32", action="store_true")
    return parser.parse_args()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_dtype(force_fp32: bool) -> torch.dtype:
    if force_fp32:
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def safe_float(x: torch.Tensor) -> torch.Tensor:
    return x.float() if x.dtype != torch.float32 else x




def build_loader(hf_dataset, tokenizer, batch_size: int, max_length: int,
                 shuffle: bool, num_workers: int) -> DataLoader:
    ds = AnswerGradingDataset(hf_dataset, tokenizer, max_length=max_length)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
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
    return compute_metrics(preds_list, labels_list)


def evaluate_all_splits(model, test_data, tokenizer, args, device: torch.device) -> Dict[str, Dict[str, float]]:
    results = {}
    for split_name in test_data.keys():
        print(f"[Eval] split={split_name}")
        results[split_name] = evaluate(model, test_data[split_name], tokenizer, args, device)
    return results




def build_model(model_name_or_dir: str, model_dtype: torch.dtype, device: torch.device,
                gradient_checkpointing: bool):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_dir,
        num_labels=2,
        dtype=torch.float32
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

def train_sft(
    model,
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

    total_steps = len(train_loader) * args.sft_epochs
    optimizer, scheduler = create_optimizer_scheduler(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        total_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
    )

    if args.selection_split not in test_data:
        raise ValueError(
            f"selection_split={args.selection_split} 不存在，当前可用：{list(test_data.keys())}"
        )

    best_score = -1.0
    best_metrics_all = None
    best_dir = run_dir / "sft_best"
    for epoch in range(1, args.sft_epochs + 1):
        model.train()
        running_loss = 0.0

        progress = tqdm(train_loader, desc=f"[SFT] Epoch {epoch}/{args.sft_epochs}")
        for batch in progress:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)

            outputs = model(**batch)

            logits = safe_float(outputs.logits)
            #loss = F.cross_entropy(logits, labels)
            loss =FocalLoss(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / max(len(train_loader), 1)
        print(f"[SFT] epoch={epoch}, avg_loss={avg_loss:.6f}")

        selection_metrics = evaluate(
            model=model,
            hf_dataset=test_data[args.selection_split],
            tokenizer=tokenizer,
            args=args,
            device=device,
        )
        print_metrics_brief(f"[SFT] {args.selection_split}", selection_metrics)

        score = selection_metrics["macro_f1"]
        if score > best_score:
            best_score = score
            print(f"[SFT] 保存新的最优 checkpoint，macro_f1={best_score:.6f}")
            save_checkpoint(model, tokenizer, best_dir)

            best_metrics_all = evaluate_all_splits(
                model=model,
                test_data=test_data,
                tokenizer=tokenizer,
                args=args,
                device=device,
            )
            save_metrics(best_metrics_all, run_dir / "sft_best_metrics")

    if best_metrics_all is None:
        # 理论上不会进入；为稳健起见保底
        save_checkpoint(model, tokenizer, best_dir)
        best_metrics_all = evaluate_all_splits(model, test_data, tokenizer, args, device)
        save_metrics(best_metrics_all, run_dir / "sft_best_metrics")

    return best_dir, best_metrics_all

# =========================
# 主流程
# =========================

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

    # 5) 阶段一：监督训练
    print("\n" + "=" * 80)
    print("[Stage 1] Supervised CE training")
    print("=" * 80)
    sft_best_dir, sft_metrics = train_sft(
        model=model,
        train_data=train_data,
        test_data=test_data,
        tokenizer=tokenizer,
        args=args,
        device=device,
        class_weights=class_weights,
        run_dir=run_dir,
    )

    print("[Stage 1] best checkpoint:", sft_best_dir)
    for split_name, metrics in sft_metrics.items():
        print_metrics_brief(f"[Stage 1][Best] {split_name}", metrics)

if __name__ == "__main__":
    main()
