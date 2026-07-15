
import os

import json

from pathlib import Path
from typing import Dict, List, Tuple, Optional



import pandas as pd

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def save_metrics(metrics_by_split: Dict[str, Dict[str, float]], save_prefix: Path):
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



