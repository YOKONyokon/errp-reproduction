# -*- coding: utf-8 -*-
"""
run_transfer.py — 跨任务迁移实验（27ch 同构三集，M4 核心）

在 HRIcursor / HRIrobot / Coadaptation 三个 27 通道同构数据集上，对比：
- 条件 raw：直接 Overlap 特征 + shrinkage sLDA；
- 条件 EA ：先做欧氏对齐（子空间域对齐），再 Overlap + sLDA。

迁移口径：
- pairwise：train A -> test B（6 个有向对）；
- LODO（留一数据集）：合并另外两个训练 -> 测试第三个（3 个）。
并报告各数据集 within-session 5 折参考，作为迁移性能的对照。
"""
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
from channel_mapping import select_channels, HOMOLOG_27, HOMOLOG_KEYS
from transfer import euclidean_align
from features import build_feature
from classifiers import build_classifier
from evaluation import balanced_accuracy
from splits import within_session_splits

# 加载并统一到 27 通道
raw = {}
for k in HOMOLOG_KEYS:
    raw[k] = select_channels(load_dataset(k), HOMOLOG_27)

# 各自做欧氏对齐（源/目标分别用自己的参考矩阵）
aligned = {}
for k in HOMOLOG_KEYS:
    Xa, _ = euclidean_align(raw[k].X)
    aligned[k] = raw[k].__class__(
        X=Xa, y=raw[k].y, ch_names=raw[k].ch_names,
        sfreq=raw[k].sfreq, subjects=raw[k].subjects)


def transfer_bacc(src, tgt):
    """Overlap + sLDA：在 src 训练、tgt 测试，返回 bacc。"""
    feat = build_feature("Overlap", src.ch_names, src.sfreq,
                         dataset_key="transfer")
    feat.fit(src.X, src.y)
    Zs, Zt = feat.transform(src.X), feat.transform(tgt.X)
    clf = build_classifier("sLDA")
    clf.fit(Zs, src.y)
    pred = clf.predict(Zt)
    return balanced_accuracy(tgt.y, pred)


def within_bacc(d):
    folds = within_session_splits(d.y, n_folds=5, seed=config.RANDOM_SEED)
    oof = np.empty_like(d.y)
    for tr, te in folds:
        feat = build_feature("Overlap", d.ch_names, d.sfreq)
        feat.fit(d.X[tr], d.y[tr])
        clf = build_classifier("sLDA")
        clf.fit(feat.transform(d.X[tr]), d.y[tr])
        oof[te] = clf.predict(feat.transform(d.X[te]))
    return balanced_accuracy(d.y, oof)


store = {"pairwise": [], "LODO": [], "within": []}

# within-session 参考
for k in HOMOLOG_KEYS:
    store["within"].append({
        "dataset": k, "raw": round(within_bacc(raw[k]), 4),
        "EA": round(within_bacc(aligned[k]), 4)})

# pairwise 迁移
for s in HOMOLOG_KEYS:
    for t in HOMOLOG_KEYS:
        if s == t:
            continue
        store["pairwise"].append({
            "source": s, "target": t,
            "raw": round(transfer_bacc(raw[s], raw[t]), 4),
            "EA": round(transfer_bacc(aligned[s], aligned[t]), 4)})

# LODO 留一数据集
for tgt in HOMOLOG_KEYS:
    src_keys = [k for k in HOMOLOG_KEYS if k != tgt]
    for cond, pool in (("raw", raw), ("EA", aligned)):
        srcX = np.concatenate([pool[k].X for k in src_keys], axis=0)
        srcy = np.concatenate([pool[k].y for k in src_keys], axis=0)
        src = pool[src_keys[0]].__class__(
            X=srcX, y=srcy, ch_names=pool[tgt].ch_names,
            sfreq=pool[tgt].sfreq,
            subjects=np.concatenate([pool[k].subjects for k in src_keys]))
        b = transfer_bacc(src, pool[tgt])
        rec = next((r for r in store["LODO"] if r["target"] == tgt), None)
        if rec is None:
            rec = {"target": tgt}
            store["LODO"].append(rec)
        rec[cond] = round(b, 4)

config.RESULTS_DIR.mkdir(exist_ok=True)
with open(config.RESULTS_DIR / "transfer.json", "w",
          encoding="utf-8") as f:
    json.dump(store, f, ensure_ascii=False, indent=2)

for name in ("within", "pairwise", "LODO"):
    print(f"--- {name} ---")
    print(pd.DataFrame(store[name]).to_string(index=False))
