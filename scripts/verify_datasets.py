# -*- coding: utf-8 -*-
"""
verify_datasets.py — 阶段 3 数据集加载核验（生成数据卡片）

对 config.ALL_DATASETS 中除 BNCI013（本阶段并行续传）外的数据集逐个加载，
校验：形状、通道、采样率、时间窗口、标签分布、被试划分、NaN 与数值范围。
输出 JSON 数据卡片，供台账与数据 pipeline 文档使用。
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

import config
from data_loader import load_dataset

# BNCI 本阶段后台续传，先核验其余 6 个
SKIP = {"BNCI013"}

# 各加载器输出单位（依据 data_loader 代码：经 MNE 的为 V，直接读 .mat 的为 µV）
UNIT = {
    "LSC": "uV", "Gaze": "uV",
    "HRIcursor": "V", "HRIrobot": "V", "Coadaptation": "V",
    "VirtualCursor": "V", "BNCI013": "V",
}

cards = []
for ds in config.ALL_DATASETS:
    key = ds["key"]
    if key in SKIP:
        continue
    print(f"[load] {key} ...", file=sys.stderr, flush=True)
    d = load_dataset(key)

    n_err = int(np.sum(d.y == config.LABEL_ERROR))
    n_cor = int(np.sum(d.y == config.LABEL_CORRECT))
    tmax = d.tmin + (d.X.shape[2] - 1) / d.sfreq
    subj, scount = np.unique(d.subjects, return_counts=True)
    lab, lcount = np.unique(d.y, return_counts=True)

    card = {
        "key": key,
        "n_trials": int(len(d.y)),
        "n_error": n_err,
        "n_correct": n_cor,
        "error_ratio": round(n_err / len(d.y), 3),
        "n_channels": int(len(d.ch_names)),
        "ch_names": list(d.ch_names),
        "sfreq": float(d.sfreq),
        "n_times": int(d.X.shape[2]),
        "tmin": float(d.tmin),
        "tmax": round(float(tmax), 3),
        "n_subjects": int(len(subj)),
        "subjects": {str(u): int(c) for u, c in zip(subj, scount)},
        "label_values": {str(u): int(c) for u, c in zip(lab, lcount)},
        "unit": UNIT.get(key, "?"),
        "dtype": str(d.X.dtype),
        "has_nan": bool(np.isnan(d.X).any()),
        "value_min": float(f"{d.X.min():.4e}"),
        "value_max": float(f"{d.X.max():.4e}"),
        "value_std": float(f"{d.X.std():.4e}"),
    }
    cards.append(card)

print(json.dumps(cards, ensure_ascii=False, indent=2))
