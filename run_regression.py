# -*- coding: utf-8 -*-
"""
run_regression.py
=================
论文 4.7“Linear regression analyses”的复现。

对每个数据集的每个被试，计算：
    r2_subject  : 选定时间窗（事件后 500ms）内、跨所有通道平均的 r^2
                  （论文原文：r^2 computed for each subject as the average
                  of r^2 values of all channels in the selected time window）；
    snr_subject : Fz 通道在该窗内的 template-based SNR（dB）；
    best_bacc   : 该被试上 Overlap 特征配 4 种分类器的最佳 balanced accuracy。

随后在全部被试上拟合两条简单线性回归：
    (1) best_bacc ~ r2_subject   论文报告 p=1.57e-5, R^2=0.241
    (2) best_bacc ~ snr_subject  论文报告 p=4.23e-4, R^2=0.150

用法：
    python3 run_regression.py                 # 全部可获取数据集
    python3 run_regression.py --datasets LSC,Gaze
输出：
    results/part2_subject_discriminability.csv
    results/part2_regression_summary.txt
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from scipy import stats

import config
from data_loader import load_dataset, EpochData
from discriminability import r2_map, template_snr
from features import build_feature
from classifiers import build_classifier
from morphology import find_channel, snr_window_for


def _light_cv_bacc(data, feature_key, clf_key, n_folds=5):
    """轻量 5 折 CV 的 balanced accuracy（不做置换检验，节省时间）。"""
    X, y = data.X, data.y
    # 某类样本不足 5 个时减少折数
    n_min = min(np.bincount(y))
    n_folds = max(2, min(n_folds, n_min))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=config.RANDOM_SEED)
    oof_true, oof_pred = np.empty_like(y), np.empty_like(y)
    for tr, te in skf.split(X, y):
        feat = build_feature(feature_key, data.ch_names, data.sfreq)
        feat.fit(X[tr], y[tr])
        Z_tr, Z_te = feat.transform(X[tr]), feat.transform(X[te])
        clf = build_classifier(clf_key)
        clf.fit(Z_tr, y[tr])
        oof_pred[te] = clf.predict(Z_te)
        oof_true[te] = y[te]
    from sklearn.metrics import balanced_accuracy_score
    return balanced_accuracy_score(oof_true, oof_pred)


def subject_row(data, subject_id, dataset_key):
    """计算单个被试的 r^2、SNR 与最佳准确率。"""
    mask = data.subjects == subject_id
    sub = EpochData(X=data.X[mask], y=data.y[mask], ch_names=data.ch_names,
                    sfreq=data.sfreq, subjects=data.subjects[mask],
                    tmin=data.tmin)
    # 单类被试无法评估，跳过
    if len(np.unique(sub.y)) < 2:
        return None

    # 论文 4.6/4.7 的数据集特定时间窗
    t0, t1 = snr_window_for(dataset_key, data.tmin, data.sfreq,
                            data.X.shape[2])

    # r^2：窗内 r2_map，再跨通道、时间平均
    r2m = r2_map(sub.X, sub.y)[:, t0:t1]
    r2_subject = float(np.mean(r2m))

    # SNR：Fz 通道、窗内、仅错误试次（论文 4.7 用 SNR of ErrP）
    ch_idx = find_channel(data.ch_names, ("Fz", "Fz"))
    if ch_idx is None:
        ch_idx = 0
    err_only = sub.X[sub.y == config.LABEL_ERROR, ch_idx, t0:t1]
    snr_lin = template_snr(err_only) if len(err_only) >= 2 else 0.0
    snr_db = 10 * np.log10(snr_lin) if snr_lin > 0 else np.nan

    # 最佳准确率：Overlap × 4 分类器
    best = max(_light_cv_bacc(sub, "Overlap", k)
               for k in ("GNB", "sLDA", "SVM", "RBF-SVM"))
    return {"subject": str(subject_id), "r2": r2_subject,
            "snr_db": snr_db, "best_bacc": best, "n_trials": len(sub.y)}


def main():
    parser = argparse.ArgumentParser(description="4.7 回归分析")
    parser.add_argument("--datasets", type=str, default=None)
    args = parser.parse_args()

    if args.datasets:
        wanted = {k.strip() for k in args.datasets.split(",")}
        datasets = [d["key"] for d in config.ALL_DATASETS if d["key"] in wanted]
    else:
        datasets = [d["key"] for d in config.ALL_DATASETS
                    if d["key"] != config.DATASET_BNCI["key"]]

    all_rows = []
    for ds_key in datasets:
        print(f"\n# 数据集：{ds_key}", flush=True)
        data = load_dataset(ds_key)
        for sid in sorted(set(data.subjects), key=str):
            t0 = time.time()
            row = subject_row(data, sid, ds_key)
            if row is None:
                continue
            row["dataset"] = ds_key
            all_rows.append(row)
            print(f"  {str(sid):6s} r2={row['r2']:.4f} snr={row['snr_db']:.1f}dB "
                  f"best={row['best_bacc']*100:.1f}% ({time.time()-t0:.1f}s)",
                  flush=True)

    df = pd.DataFrame(all_rows)
    df = df[["dataset", "subject", "r2", "snr_db", "best_bacc", "n_trials"]]
    csv_path = config.RESULTS_DIR / "part2_subject_discriminability.csv"
    df.to_csv(csv_path, index=False)

    # 两条线性回归
    lines = ["4.7 线性回归分析（逐被试，n=%d）" % len(df), "=" * 50]

    def _reg(x, y, name):
        m = stats.linregress(x, y)
        lines.append(
            f"\n[{name}]\n  slope={m.slope:.4f} intercept={m.intercept:.4f}\n"
            f"  R^2={m.rvalue**2:.3f}  p={m.pvalue:.3e}  n={len(x)}")
        return m

    _reg(df["r2"].values, df["best_bacc"].values, "best_bacc ~ r^2")
    valid = df.dropna(subset=["snr_db"])
    _reg(valid["snr_db"].values, valid["best_bacc"].values,
         "best_bacc ~ SNR(dB)")

    txt = "\n".join(lines)
    print("\n" + txt)
    txt_path = config.RESULTS_DIR / "part2_regression_summary.txt"
    txt_path.write_text(txt, encoding="utf-8")
    print("\n已保存：", csv_path, "\n        ", txt_path)


if __name__ == "__main__":
    main()
