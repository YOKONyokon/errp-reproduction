# -*- coding: utf-8 -*-
"""
run_experiments.py
==================
论文第 4 节“受控对比实验”（4.2–4.5）的运行入口。

对每个数据集，用 sLDA 作为主分类器，对“无处理基线 vs 有处理”做配对比较，
输出每个实验条件的 out-of-fold balanced accuracy：

    4.2 Feature selection : baseline / r^2 选择 / PCA 选择
    4.3 Time window       : baseline / 判别力时间窗
    4.4 Augmentation      : baseline / 双时间窗 AND 决策
    4.5 Downsampling      : baseline / 窗口平均降采样

用法：
    python3 run_experiments.py                          # 全部可获取数据集，Overlap 特征
    python3 run_experiments.py --datasets LSC,Gaze      # 只跑指定数据集
    python3 run_experiments.py --features Overlap,FCzCz # 指定特征
    python3 run_experiments.py --quick                  # 快速冒烟（每集 1 被试）

输出：
    results/part2_conditions.csv   数据集×特征×条件的 bacc 配对表
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

import config
from data_loader import load_dataset
from experiments import run_condition

# 条件 key -> 论文小节
CONDITIONS = [
    ("baseline", "基线"),
    ("fs_r2", "4.2 r^2 选择"),
    ("fs_pca", "4.2 PCA 选择"),
    ("window", "4.3 时间窗"),
    ("augment", "4.4 增强"),
    ("downsample", "4.5 降采样"),
]


def main():
    parser = argparse.ArgumentParser(description="第 4 节受控对比实验")
    parser.add_argument("--datasets", type=str, default=None,
                        help="逗号分隔数据集 key，默认全部可获取")
    parser.add_argument("--features", type=str, default="Overlap",
                        help="逗号分隔特征 key，默认 Overlap")
    parser.add_argument("--quick", action="store_true",
                        help="快速冒烟：多被试数据集只用前 1 个被试")
    args = parser.parse_args()

    config.ensure_dirs()

    if args.datasets:
        wanted = {k.strip() for k in args.datasets.split(",") if k.strip()}
        datasets = [d["key"] for d in config.ALL_DATASETS if d["key"] in wanted]
    else:
        # BNCI 完整数据未下载完，自动排除
        datasets = [d["key"] for d in config.ALL_DATASETS
                    if d["key"] != config.DATASET_BNCI["key"]]
    features = [k.strip() for k in args.features.split(",") if k.strip()]

    rows = []
    for ds_key in datasets:
        max_sub = 1 if args.quick else None
        print(f"\n{'#'*60}\n# 加载数据集：{ds_key}\n{'#'*60}", flush=True)
        data = load_dataset(ds_key, max_subjects=max_sub)
        print(data.summary(), flush=True)

        for feat in features:
            for cond, label in CONDITIONS:
                t0 = time.time()
                bacc = run_condition(data, feat, "sLDA", cond)
                print(f"  {feat:9s} {label:14s} bacc={bacc*100:6.2f}% "
                      f"({time.time()-t0:.1f}s)", flush=True)
                rows.append({"dataset": ds_key, "feature": feat,
                             "condition": cond, "condition_label": label,
                             "bacc": round(bacc, 4),
                             "bacc_pct": round(bacc * 100, 2)})

    out = pd.DataFrame(rows)
    save_path = config.RESULTS_DIR / "part2_conditions.csv"
    out.to_csv(save_path, index=False)

    # 控制台打印配对透视表
    print("\n" + "=" * 64)
    print("配对实验 balanced accuracy（%）")
    print("=" * 64)
    piv = out.pivot_table(index=["dataset", "feature"],
                          columns="condition", values="bacc_pct")
    print(piv.round(2).to_string())
    print("\n结果已保存：", save_path)


if __name__ == "__main__":
    main()
