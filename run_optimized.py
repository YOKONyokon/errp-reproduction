# -*- coding: utf-8 -*-
"""
run_optimized.py
================
只重跑受优化影响的空间滤波特征（xDAWN、FCB）共 8 条管线/数据集，
就地更新各数据集的 bacc_matrix / detailed / heatmap，并重建
summary_all.csv 与 comparison_paper_vs_mine.csv。

非空间特征（FCzCz、Timepoints、Overlap）的实现未改动，其结果直接复用，
从而在 2 核机器上把重跑成本从 120 条降到 48 条。

用法：
    python3 run_optimized.py                 # 全部 6 数据集，1000 次置换
    python3 run_optimized.py --quick         # 50 次置换，快速验证
    python3 run_optimized.py --datasets LSC,Gaze
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

import config
from data_loader import load_dataset
from evaluation import evaluate_pipeline
from plotting import plot_bacc_heatmap

REOPT_FEATURES = ["xDAWN", "FCB"]


def update_one_dataset(ds_key, n_perm):
    """重跑某数据集的 xDAWN/FCB 并更新结果文件。"""
    print(f"\n{'#'*64}\n# {ds_key}：重跑 xDAWN/FCB\n{'#'*64}", flush=True)
    data = load_dataset(ds_key)

    matrix_path = config.RESULTS_DIR / f"{ds_key}_bacc_matrix.csv"
    detailed_path = config.RESULTS_DIR / f"{ds_key}_detailed.csv"
    matrix = pd.read_csv(matrix_path, index_col=0)
    detailed = pd.read_csv(detailed_path)

    for fe in REOPT_FEATURES:
        for clf in config.CLASSIFIER_KEYS:
            t0 = time.time()
            res = evaluate_pipeline(data, fe, clf, n_perm=n_perm,
                                    dataset_key=ds_key)
            bacc = res["bacc"]
            matrix.loc[fe, clf] = bacc
            # 替换 detailed 对应行
            msk = (detailed["feature"] == fe) & \
                  (detailed["classifier"] == clf)
            new_row = {
                "dataset": ds_key, "feature": fe, "classifier": clf,
                "bacc": round(bacc, 4), "tpr": round(res["tpr"], 4),
                "fpr": round(res["fpr"], 4),
                "chance95": round(res["permutation"]["chance_level_95"], 4),
                "p_value": res["permutation"]["p_value"]}
            if msk.any():
                for k, v in new_row.items():
                    detailed.loc[msk, k] = v
            else:
                detailed = pd.concat(
                    [detailed, pd.DataFrame([new_row])], ignore_index=True)
            print(f"  {fe:5s}+{clf:7s} bacc={bacc*100:6.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # 保存
    matrix.to_csv(matrix_path)
    detailed.to_csv(detailed_path, index=False)
    # 重新生成热力图
    plot_bacc_heatmap(
        matrix.values, list(matrix.index), list(matrix.columns),
        f"{ds_key} balanced accuracy (%)",
        config.RESULTS_DIR / f"{ds_key}_bacc_heatmap.png")
    return matrix


def rebuild_summary_and_comparison():
    """从各数据集 detailed 重建 summary，并用旧 paper_pct 重建 comparison。"""
    old_cmp = pd.read_csv("comparison_paper_vs_mine.csv")
    paper_map = {(r["dataset"], r["feature"], r["classifier"]): r["paper_pct"]
                 for _, r in old_cmp.iterrows()}

    detail_files = sorted(config.RESULTS_DIR.glob("*_detailed.csv"))
    frames = [pd.read_csv(f) for f in detail_files]
    summary = pd.concat(frames, ignore_index=True)
    summary = summary[summary["dataset"].isin(
        [d["key"] for d in config.ALL_DATASETS])]
    summary.to_csv(config.RESULTS_DIR / "summary_all.csv", index=False)

    # 重建 comparison
    rows = []
    for _, r in summary.iterrows():
        key = (r["dataset"], r["feature"], r["classifier"])
        paper = paper_map.get(key, np.nan)
        my = r["bacc"] * 100
        diff = (paper - my) if not pd.isna(paper) else np.nan
        rows.append({"dataset": r["dataset"], "feature": r["feature"],
                     "classifier": r["classifier"], "paper_pct": paper,
                     "my_pct": round(my, 2),
                     "diff": round(diff, 2) if not pd.isna(diff) else np.nan})
    cmp = pd.DataFrame(rows)
    cmp.to_csv("comparison_paper_vs_mine.csv", index=False)
    return cmp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--datasets", type=str, default=None)
    args = parser.parse_args()
    n_perm = 50 if args.quick else config.N_PERMUTATIONS

    if args.datasets:
        wanted = {k.strip() for k in args.datasets.split(",")}
        datasets = [d["key"] for d in config.ALL_DATASETS
                    if d["key"] in wanted and d["key"] != "BNCI013"]
    else:
        datasets = [d["key"] for d in config.ALL_DATASETS
                    if d["key"] != "BNCI013"]

    for ds in datasets:
        update_one_dataset(ds, n_perm)

    cmp = rebuild_summary_and_comparison()
    c = cmp.dropna(subset=["paper_pct", "my_pct"])
    print("\n" + "=" * 60)
    print("优化后整体差异（paper - mine）")
    print("=" * 60)
    print("MAE:", round(c["diff"].abs().mean(), 2),
          " mean diff:", round(c["diff"].mean(), 2),
          " within5:", (c["diff"].abs() <= 5).sum(), "/", len(c))
    print("\n按特征：")
    print(c.groupby("feature")["diff"].agg(["mean", "count"]).round(2))
    print("\n按分类器：")
    print(c.groupby("classifier")["diff"].agg(["mean", "count"]).round(2))


if __name__ == "__main__":
    main()
