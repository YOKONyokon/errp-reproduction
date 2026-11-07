# -*- coding: utf-8 -*-
"""
run_morphology.py
=================
论文 4.6“ErrP morphology, SNR and discrimination”的运行入口。

对每个可获取数据集，在 Fz 通道（缺失时退化）上提取错误试次 ErrP 的
形态学指标：第一负峰幅度/潜伏期、正峰幅度/潜伏期，以及错误与正确两类
在论文选定时间窗内的 template-based SNR（dB）。

用法：
    python3 run_morphology.py
    python3 run_morphology.py --datasets LSC,Gaze
输出：
    results/part2_morphology.csv
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

import config
from data_loader import load_dataset
from morphology import dataset_morphology


def main():
    parser = argparse.ArgumentParser(description="4.6 形态学分析")
    parser.add_argument("--datasets", type=str, default=None)
    args = parser.parse_args()

    if args.datasets:
        wanted = {k.strip() for k in args.datasets.split(",")}
        datasets = [d["key"] for d in config.ALL_DATASETS if d["key"] in wanted]
    else:
        datasets = [d["key"] for d in config.ALL_DATASETS
                    if d["key"] != config.DATASET_BNCI["key"]]

    rows = []
    for ds_key in datasets:
        print(f"# 数据集：{ds_key}", flush=True)
        data = load_dataset(ds_key)
        m = dataset_morphology(data, dataset_key=ds_key)
        m["dataset"] = ds_key
        rows.append(m)
        print(f"  通道={m['channel']} 负峰={m['neg_amp']:.2f}µV@{m['neg_latency']*1000:.0f}ms "
              f"正峰={m['pos_amp']:.2f}µV@{m['pos_latency']*1000:.0f}ms "
              f"SNR(错)={m['snr_err_db']:.1f}dB SNR(正)={m['snr_cor_db']:.1f}dB",
              flush=True)

    df = pd.DataFrame(rows)
    cols = ["dataset", "channel", "neg_amp", "neg_latency", "pos_amp",
            "pos_latency", "snr_err_db", "snr_cor_db"]
    df = df[cols]
    save = config.RESULTS_DIR / "part2_morphology.csv"
    df.to_csv(save, index=False)

    print("\n" + "=" * 70)
    print("4.6 形态学指标（幅度 µV，潜伏期 s，SNR dB）")
    print("=" * 70)
    print(df.round(3).to_string(index=False))
    print("\n已保存：", save)


if __name__ == "__main__":
    main()
