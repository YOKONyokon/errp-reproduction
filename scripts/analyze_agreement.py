# -*- coding: utf-8 -*-
"""
analyze_agreement.py — 基线复现吻合度分析（M3 量化证据）

读取 comparison_paper_vs_mine.csv，对论文报告了数值的对比单元统计：
MAE、有符号平均偏差、中位偏差、差≤5/≤10 占比，并逐数据集/特征/分类器分解。
结果同时打印并写入 results/agreement_summary.json。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import config

df = pd.read_csv(ROOT / "comparison_paper_vs_mine.csv")
has = df.dropna(subset=["paper_pct"]).copy()
abs_diff = has["diff"].abs()

summary = {
    "total_pipelines": int(len(df)),
    "cells_with_paper_value": int(len(has)),
    "MAE": round(float(abs_diff.mean()), 2),
    "mean_signed_diff": round(float(has["diff"].mean()), 2),
    "median_abs_diff": round(float(abs_diff.median()), 2),
    "pct_within_5": round(float((abs_diff <= 5).mean() * 100), 1),
    "pct_within_10": round(float((abs_diff <= 10).mean() * 100), 1),
    "MAE_by_dataset": {
        k: round(float(v), 2)
        for k, v in has.groupby("dataset")["diff"]
        .apply(lambda s: s.abs().mean()).items()},
    "MAE_by_feature": {
        k: round(float(v), 2)
        for k, v in has.groupby("feature")["diff"]
        .apply(lambda s: s.abs().mean()).items()},
    "MAE_by_classifier": {
        k: round(float(v), 2)
        for k, v in has.groupby("classifier")["diff"]
        .apply(lambda s: s.abs().mean()).items()},
}

config.RESULTS_DIR.mkdir(exist_ok=True)
with open(config.RESULTS_DIR / "agreement_summary.json", "w",
          encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
