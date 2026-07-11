import os
import os

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def compute_metrics(preds: List[int], labels: List[int]) -> Dict[str, float]:
    preds_np = np.asarray(preds, dtype=np.int64)
    labels_np = np.asarray(labels, dtype=np.int64)

    if len(labels_np) == 0:
        return {}

    accuracy = float((preds_np == labels_np).mean())

    # 约定 positive=correct(1)
    tp = int(((preds_np == 1) & (labels_np == 1)).sum())
    tn = int(((preds_np == 0) & (labels_np == 0)).sum())
    fp = int(((preds_np == 1) & (labels_np == 0)).sum())
    fn = int(((preds_np == 0) & (labels_np == 1)).sum())

    precision_pos = _safe_div(tp, tp + fp)
    recall_pos = _safe_div(tp, tp + fn)
    f1_pos = _safe_div(2 * precision_pos * recall_pos, precision_pos + recall_pos)

    precision_neg = _safe_div(tn, tn + fn)
    recall_neg = _safe_div(tn, tn + fp)
    f1_neg = _safe_div(2 * precision_neg * recall_neg, precision_neg + recall_neg)

    macro_precision = (precision_pos + precision_neg) / 2.0
    macro_recall = (recall_pos + recall_neg) / 2.0
    macro_f1 = (f1_pos + f1_neg) / 2.0

    support_pos = int((labels_np == 1).sum())
    support_neg = int((labels_np == 0).sum())
    n = support_pos + support_neg
    weighted_f1 = _safe_div(f1_pos * support_pos + f1_neg * support_neg, n)

    # Cohen's kappa
    pred_pos = int((preds_np == 1).sum())
    pred_neg = int((preds_np == 0).sum())
    p0 = accuracy
    pe = _safe_div(support_pos * pred_pos + support_neg * pred_neg, n * n)
    kappa = _safe_div(p0 - pe, 1.0 - pe) if (1.0 - pe) != 0 else 0.0

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "precision_correct_1": precision_pos,
        "recall_correct_1": recall_pos,
        "f1_correct_1": f1_pos,
        "precision_incorrect_0": precision_neg,
        "recall_incorrect_0": recall_neg,
        "f1_incorrect_0": f1_neg,
        "cohen_kappa": kappa,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "support_correct_1": support_pos,
        "support_incorrect_0": support_neg,
        "num_samples": n,
    }

