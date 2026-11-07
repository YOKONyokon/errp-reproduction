# -*- coding: utf-8 -*-
"""
run_learning_curve.py — 小样本学习曲线（M4）

固定一个分层 20% 测试集，从训练池按比例 [5/10/25/50/100%] 分层抽样训练，
每个比例用 3 个随机种子，在同一测试集上评估 bacc，刻画 bacc 随训练样本量
的变化。默认经典 Overlap+sLDA（快速，覆盖 6 集）；--model EEGNet 时在
指定数据集上做深度学习曲线（CPU 慢，建议单集）。
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import config
from data_loader import load_dataset
from features import build_feature
from classifiers import build_classifier
from evaluation import balanced_accuracy
from deep_training import train_eegnet

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="sLDA", choices=["sLDA", "EEGNet"])
parser.add_argument("--datasets", default="")
args = parser.parse_args()
only = [s for s in args.datasets.split(",") if s]

FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.0]
SEEDS = [42, 123, 2024]
records = []

for ds in config.ALL_DATASETS:
    key = ds["key"]
    if key in ("BNCI013", "ERPCORE_ERN") or (only and key not in only):
        continue
    data = load_dataset(key)
    # 固定 20% 测试集
    tr_pool, te = next(StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=config.RANDOM_SEED)
        .split(data.X, data.y))
    Xte, yte = data.X[te], data.y[te]

    for frac in FRACTIONS:
        n_keep = max(2, int(round(len(tr_pool) * frac)))
        baccs = []
        for sd in SEEDS:
            if n_keep >= len(tr_pool):
                # 全量训练，不再抽样（StratifiedShuffleSplit 不允许 train_size=全集）
                idx = tr_pool
            else:
                sub, _ = next(StratifiedShuffleSplit(
                    n_splits=1, train_size=n_keep, random_state=sd)
                    .split(data.X[tr_pool], data.y[tr_pool]))
                idx = tr_pool[sub]
            Xtr, ytr = data.X[idx], data.y[idx]
            if args.model == "sLDA":
                feat = build_feature("Overlap", data.ch_names, data.sfreq)
                feat.fit(Xtr, ytr)
                clf = build_classifier("sLDA")
                clf.fit(feat.transform(Xtr), ytr)
                pred = clf.predict(feat.transform(Xte))
            else:
                pred, _ = train_eegnet(
                    Xtr, ytr, Xte, yte,
                    data.X.shape[1], data.X.shape[2], seed=sd)
            baccs.append(balanced_accuracy(yte, pred))
        records.append({
            "dataset": key, "model": args.model,
            "fraction": frac, "n_train": n_keep,
            "bacc_mean": round(float(np.mean(baccs)), 4),
            "bacc_std": round(float(np.std(baccs)), 4)})
    print(f"{key:13s} done", flush=True)

config.RESULTS_DIR.mkdir(exist_ok=True)
out = pd.DataFrame(records)
fname = f"learning_curve_{args.model}.csv"
out.to_csv(config.RESULTS_DIR / fname, index=False)
with open(config.RESULTS_DIR / f"learning_curve_{args.model}.json", "w",
          encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)
print("saved", fname)
