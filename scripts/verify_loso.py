# -*- coding: utf-8 -*-
"""
verify_loso.py — LOSO 跨被试端到端验证

在指定多被试数据集上用留一被试（LOSO）协议跑完整管线，逐被试报告 bacc，
验证新固化的 splits.loso_splits 在真实特征/分类下可用，且每折被试完全隔离。
默认 VirtualCursor（8 被试、16 通道，规模适中）。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from data_loader import load_dataset
from splits import loso_splits, assert_group_disjoint
from features import build_feature
from classifiers import build_classifier
from evaluation import balanced_accuracy

parser = argparse.ArgumentParser(description="LOSO 端到端验证")
parser.add_argument("--dataset", default="VirtualCursor")
parser.add_argument("--feature", default="Overlap")
parser.add_argument("--clf", default="sLDA")
args = parser.parse_args()

data = load_dataset(args.dataset)
folds = loso_splits(data.subjects)
print(f"dataset={args.dataset} subjects={len(folds)} "
 f"feature={args.feature} clf={args.clf}")

per_subject = []
for tr, te in folds:
    assert assert_group_disjoint(tr, te, data.subjects), "LOSO 泄漏！"
    test_subj = sorted(set(data.subjects[te].tolist()))[0]
    feat = build_feature(args.feature, data.ch_names, data.sfreq,
                         dataset_key=args.dataset)
    feat.fit(data.X[tr], data.y[tr])
    Ztr, Zte = feat.transform(data.X[tr]), feat.transform(data.X[te])
    clf = build_classifier(args.clf)
    clf.fit(Ztr, data.y[tr])
    pred = clf.predict(Zte)
    b = balanced_accuracy(data.y[te], pred)
    per_subject.append(b)
    print(f"  held-out {test_subj:8s} n={len(te):4d} bacc={b:.3f}")

print(f"LOSO mean bacc = {np.mean(per_subject):.3f} "
      f"(std {np.std(per_subject):.3f})")
