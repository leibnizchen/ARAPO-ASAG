


import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from datasets import load_dataset, ClassLabel

try:
    import pandas as pd
except Exception:
    pd = None

import torch
import torch.nn.functional as F






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


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# =========================
# 数据加载与标签映射
# =========================

def load_raw_dataset(dataset_name: str):
    """
    保留 baseline 的数据路径与测试集组织方式。
    """
    if dataset_name == "beetle":
        dataset = load_dataset("DataSet/beetle")
        train_data = dataset["train"]
        test_data = load_dataset(
            "parquet",
            data_files={
                "test_ua": "DataSet/beetle/test-ua-00001.parquet",
                "test_uq": "DataSet/beetle/test-uq-00001.parquet",
            }
        )
    elif dataset_name == "scientsbank":
        dataset = load_dataset("DataSet/scientsbank")
        train_data = dataset["train"]
        test_data = load_dataset(
            "parquet",
            data_files={
                "test_ua": "DataSet/scientsbank/test-ua-00001.parquet",
                "test_ud": "DataSet/scientsbank/test-ud-00001.parquet",
                "test_uq": "DataSet/scientsbank/test-uq-00001.parquet",
            }
        )
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    return train_data, test_data


def align_binary_labels(data_obj):
    """
    与原 baseline 保持一致：
    原标签 correct / contradictory / partially_correct_incomplete / irrelevant / non_domain
    -> 二分类 correct / incorrect
    -> 最终 incorrect=0, correct=1
    """
    data_obj = data_obj.align_labels_with_mapping(
        {
            "correct": 0,
            "contradictory": 1,
            "partially_correct_incomplete": 1,
            "irrelevant": 1,
            "non_domain": 1,
        },
        "label",
    )
    data_obj = data_obj.cast_column("label", ClassLabel(names=["correct", "incorrect"]))
    data_obj = data_obj.align_labels_with_mapping(
        {"incorrect": 0, "correct": 1},
        "label",
    )
    return data_obj


def label_distribution(hf_dataset) -> Dict[str, int]:
    labels = [int(x["label"]) for x in hf_dataset]
    bincount = np.bincount(labels, minlength=2)
    return {"incorrect_0": int(bincount[0]), "correct_1": int(bincount[1])}


class AnswerGradingDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length: int = 512):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = (
            f"Question: {item['question']}\n"
            f"Reference: {item['reference_answer']}\n"
            f"Student: {item['student_answer']}\n"
        )

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["labels"] = torch.tensor(int(item["label"]), dtype=torch.long)
        return encoding


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


def compute_class_weights(train_data, device: torch.device,
                          disable_class_weights: bool) -> Optional[torch.Tensor]:
    if disable_class_weights:
        return None

    labels = np.array([int(x["label"]) for x in train_data], dtype=np.int64)
    counts = np.bincount(labels, minlength=2).astype(np.float64)

    # 避免某类极端缺失造成除零
    counts = np.maximum(counts, 1.0)
    total = counts.sum()
    weights = total / (2.0 * counts)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)

    return weights

