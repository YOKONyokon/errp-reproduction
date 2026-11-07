# -*- coding: utf-8 -*-
"""
smoke_pipeline.py — 已有数据集端到端冒烟验证

在 6 个已下载数据集上，用固化的 within-session 5 折划分，跑一条轻量但完整的
管线（Overlap 特征 + sLDA），验证：加载 → 划分 → 折内 fit 特征/分类器 →
oof 预测 → balanced accuracy 全链路可用，且结果显著高于 chance（0.5）。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

import config
from data_loader import load_dataset
from splits import within_session_splits
from features import build_feature
from classifiers import build_classifier
from evaluation import balanced_accuracy

# 与正式 pipeline 一致的轻量组合
FEAT_KEY, CLF_KEY = "Overlap", "sLDA"
SKIP = {"BNCI013", "ERPCORE_ERN"}

print(f"{'dataset':13s} {'n':>5s} {'folds':>5s} {'bacc':>6s}  non-chance")
for ds in config.ALL_DATASETS:
    key = ds["key"]
    if key in SKIP:
        continue
    data = load_dataset(key)
    folds = within_session_splits(data.y, n_folds=5, seed=config.RANDOM_SEED)
    oof_pred = np.empty_like(data.y)
    for tr, te in folds:
        feat = build_feature(FEAT_KEY, data.ch_names, data.sfreq,
                             dataset_key=key)
        feat.fit(data.X[tr], data.y[tr])
        Ztr = feat.transform(data.X[tr])
        Zte = feat.transform(data.X[te])
        clf = build_classifier(CLF_KEY)
        clf.fit(Ztr, data.y[tr])
        oof_pred[te] = clf.predict(Zte)
    bacc = balanced_accuracy(data.y, oof_pred)
    print(f"{key:13s} {len(data.y):5d} {len(folds):5d} "
          f"{bacc:6.3f}  {bacc > 0.5}")
