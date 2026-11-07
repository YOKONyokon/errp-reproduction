# -*- coding: utf-8 -*-
"""
run_deep_baseline.py — EEGNet 深度基线（6 集 within-session，多随机种子）

在 6 个已下载数据集上，用 within-session 分层 5 折 + 多个随机种子训练 EEGNet，
记录每集 bacc 的均值与标准差，输出 results/deep_baseline.csv 与 JSON。
用法：
    python run_deep_baseline.py                 # 全量 3 种子
    python run_deep_baseline.py --seeds 42      # 单种子快速
    python run_deep_baseline.py --datasets LSC,Gaze
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

import config
from data_loader import load_dataset
from splits import within_session_splits
from deep_training import train_eegnet
from evaluation import balanced_accuracy

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", default="42,123,2024")
parser.add_argument("--datasets", default="")
args = parser.parse_args()
seeds = [int(s) for s in args.seeds.split(",")]
only = [s for s in args.datasets.split(",") if s]

rows = []
for ds in config.ALL_DATASETS:
    key = ds["key"]
    if key in ("BNCI013", "ERPCORE_ERN") or (only and key not in only):
        continue
    data = load_dataset(key)
    n_ch, n_t = data.X.shape[1], data.X.shape[2]
    seed_bacc = []
    t0 = time.time()
    for sd in seeds:
        folds = within_session_splits(data.y, n_folds=5, seed=sd)
        oof = np.empty_like(data.y)
        for tr, te in folds:
            pred, _ = train_eegnet(data.X[tr], data.y[tr],
                                   data.X[te], data.y[te],
                                   n_ch, n_t, seed=sd)
            oof[te] = pred
        seed_bacc.append(balanced_accuracy(data.y, oof))
    rows.append({
        "dataset": key,
        "model": "EEGNet",
        "n_trials": len(data.y),
        "bacc_mean": round(float(np.mean(seed_bacc)), 4),
        "bacc_std": round(float(np.std(seed_bacc)), 4),
        "seeds": ";".join(map(str, seeds)),
        "seconds": round(time.time() - t0, 1),
    })
    print(f"{key:13s} bacc={np.mean(seed_bacc):.3f}±{np.std(seed_bacc):.3f} "
          f"({rows[-1]['seconds']}s)", flush=True)

out = pd.DataFrame(rows)
config.RESULTS_DIR.mkdir(exist_ok=True)
out.to_csv(config.RESULTS_DIR / "deep_baseline.csv", index=False)
with open(config.RESULTS_DIR / "deep_baseline.json", "w",
          encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print("saved results/deep_baseline.csv")
