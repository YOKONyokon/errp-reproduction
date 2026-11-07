# -*- coding: utf-8 -*-
"""
run_reproduction.py
===================
ERRP 复现主入口：在 6 个可获取真实数据集上，逐一运行“5 特征 × 4 分类器”
共 20 条管线，保存 balanced accuracy 结果表、JSON、热力图、混淆矩阵与平均波形。

用法：
    python3 run_reproduction.py                 # 默认全量
    python3 run_reproduction.py --hri-subjects 3   # 多被试数据集只用前 3 个
    python3 run_reproduction.py --quick         # 快速冒烟（每集 1 被试、少置换）
    python3 run_reproduction.py --datasets HRIrobot,Coadaptation  # 只跑指定数据集

断点续跑（checkpoint）：
    每跑完一条管线，立即把结果写入 {dataset}_partial.csv / .json。
    若进程被中断，重新运行同一命令会自动跳过已完成的管线，从断点继续。
    全部 20 条完成后才写正式结果文件并清理 partial。

输出（results/ 目录）：
    {dataset}_bacc_matrix.csv   特征×分类器 bacc 矩阵
    {dataset}_detailed.csv      每条管线的 TPR/FPR/p/chance 明细
    {dataset}_results.json      完整结果（含混淆矩阵、置换检验）
    {dataset}_bacc_heatmap.png  bacc 热力图
    {dataset}_waveform.png      错误 vs 正确平均波形
    summary_all.csv             跨数据集汇总
"""

import argparse
import json
import sys
import time
from pathlib import Path

# 把 src 与项目根加入路径
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

import config
from data_loader import load_dataset
from evaluation import evaluate_pipeline
import plotting


def _checkpoint_paths(dataset_key):
    """返回某数据集的两个增量 checkpoint 文件路径（明细 CSV 与完整 JSON）。"""
    return (config.RESULTS_DIR / f"{dataset_key}_partial.csv",
            config.RESULTS_DIR / f"{dataset_key}_partial.json")


def _load_checkpoint(dataset_key):
    """
    读取已有 checkpoint，恢复已完成管线。

    返回：
        done_map : {(feature, classifier): 一行明细 dict}
        full_json: {“feature+classifier”: 含混淆矩阵的完整结果 dict}
    若没有 checkpoint，则返回两个空容器。
    """
    csv_path, json_path = _checkpoint_paths(dataset_key)
    done_map, full_json = {}, {}
    if csv_path.exists():
        prev = pd.read_csv(csv_path)
        for _, row in prev.iterrows():
            done_map[(row["feature"], row["classifier"])] = row.to_dict()
    if json_path.exists():
        with open(json_path, "r") as f:
            full_json = json.load(f)
    return done_map, full_json


def _save_checkpoint(dataset_key, records, full_json):
    """把当前已完成的全部管线增量落盘（CSV + JSON），保证中断不丢进度。"""
    csv_path, json_path = _checkpoint_paths(dataset_key)
    pd.DataFrame(records).to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(full_json, f, ensure_ascii=False, indent=2)


def run_one_dataset(dataset_key, hri_subjects, n_perm):
    """对单个数据集跑完整方法矩阵并保存全部结果（支持断点续跑）。"""
    print(f"\n{'#'*64}\n# 数据集：{dataset_key}\n{'#'*64}", flush=True)

    # 1) 加载数据
    # LSC 已预切且无独立被试概念；其余多被试数据集都支持 max_subjects
    if dataset_key == config.DATASET_LSC["key"]:
        max_sub = None
    else:
        max_sub = hri_subjects
    data = load_dataset(dataset_key, max_subjects=max_sub)
    print("数据摘要：", data.summary(), flush=True)

    # 2) 恢复 checkpoint（若此前已跑过一部分）
    done_map, full_json = _load_checkpoint(dataset_key)
    if done_map:
        print(f"检测到 checkpoint：已完成 {len(done_map)} 条管线，跳过它们。",
              flush=True)

    records = []
    bacc_matrix = np.zeros((len(config.FEATURE_KEYS),
                            len(config.CLASSIFIER_KEYS)))

    # 3) 逐管线评估（已完成的直接从 checkpoint 取值）
    for i, feat_key in enumerate(config.FEATURE_KEYS):
        for j, clf_key in enumerate(config.CLASSIFIER_KEYS):
            pipe_id = (feat_key, clf_key)

            if pipe_id in done_map:
                # 断点续跑：复用已保存结果，不重复计算
                row = done_map[pipe_id]
                bacc_matrix[i, j] = row["bacc"]
                records.append({k: row[k] for k in
                                ("dataset", "feature", "classifier", "bacc",
                                 "tpr", "fpr", "chance95", "p_value")})
                print(f"  -> 特征 {feat_key} × 分类器 {clf_key}："
                      f"已存在 bacc={row['bacc']:.4f}，跳过", flush=True)
                continue

            print(f"  -> 特征 {feat_key} × 分类器 {clf_key} ...", flush=True)
            _t0 = time.time()
            res = evaluate_pipeline(data, feat_key, clf_key, n_perm=n_perm,
                                    dataset_key=dataset_key)
            print(f"     完成，bacc={res['bacc']:.4f}，"
                  f"耗时 {time.time()-_t0:.1f}s", flush=True)

            bacc_matrix[i, j] = res["bacc"]
            records.append({
                "dataset": dataset_key,
                "feature": feat_key,
                "classifier": clf_key,
                "bacc": round(res["bacc"], 4),
                "tpr": round(res["tpr"], 4),
                "fpr": round(res["fpr"], 4),
                "chance95": round(res["permutation"]["chance_level_95"], 4),
                "p_value": res["permutation"]["p_value"],
            })
            # JSON 里保留完整结构（含混淆矩阵，供出图使用）
            full_json[f"{feat_key}+{clf_key}"] = {
                "bacc": res["bacc"], "tpr": res["tpr"], "fpr": res["fpr"],
                "confusion_matrix": res["confusion_matrix"],
                "permutation": res["permutation"],
            }

            # 每条管线完成后立即落盘，确保随时中断都能保留进度
            _save_checkpoint(dataset_key, records, full_json)

    # 4) 全部 20 条完成，保存正式表格
    matrix_df = pd.DataFrame(bacc_matrix, index=config.FEATURE_KEYS,
                             columns=config.CLASSIFIER_KEYS)
    matrix_df.to_csv(config.RESULTS_DIR / f"{dataset_key}_bacc_matrix.csv")
    detail_df = pd.DataFrame(records)
    detail_df.to_csv(config.RESULTS_DIR / f"{dataset_key}_detailed.csv",
                     index=False)
    with open(config.RESULTS_DIR / f"{dataset_key}_results.json", "w") as f:
        json.dump(full_json, f, ensure_ascii=False, indent=2)

    # 5) 出图
    plotting.plot_bacc_heatmap(
        bacc_matrix, config.FEATURE_KEYS, config.CLASSIFIER_KEYS,
        f"{dataset_key} · Balanced accuracy（%）",
        config.RESULTS_DIR / f"{dataset_key}_bacc_heatmap.png")
    plotting.plot_waveforms(
        data, config.RESULTS_DIR / f"{dataset_key}_waveform.png", dataset_key)

    # 代表性管线（最佳与 FCB+sLDA）的混淆矩阵
    best = detail_df.loc[detail_df["bacc"].idxmax()]
    best_key = (best["feature"], best["classifier"])
    for tag, (fk, ck) in {"best": best_key,
                          "FCB_sLDA": ("FCB", "sLDA")}.items():
        cm = full_json[f"{fk}+{ck}"]["confusion_matrix"]
        plotting.plot_confusion(
            cm, ["正确", "错误"], f"{dataset_key} · {fk}+{ck}",
            config.RESULTS_DIR / f"{dataset_key}_confusion_{tag}.png")

    # 6) 正式结果已写好，清理 checkpoint
    for p in _checkpoint_paths(dataset_key):
        if p.exists():
            p.unlink()

    return detail_df


def main():
    parser = argparse.ArgumentParser(description="ERRP 复现主入口")
    parser.add_argument("--hri-subjects", type=int, default=None,
                        help="多被试数据集使用的被试数量（默认全部）")
    parser.add_argument("--datasets", type=str, default=None,
                        help="逗号分隔的数据集 key，默认全部（如 HRIrobot,VirtualCursor）")
    parser.add_argument("--quick", action="store_true",
                        help="快速冒烟：每集 1 被试、100 次置换")
    args = parser.parse_args()

    config.ensure_dirs()
    hri_subjects = 1 if args.quick else args.hri_subjects
    n_perm = 100 if args.quick else config.N_PERMUTATIONS

    # 按 --datasets 过滤要运行的数据集（默认全部）
    if args.datasets:
        wanted = {k.strip() for k in args.datasets.split(",") if k.strip()}
        selected = [d for d in config.ALL_DATASETS if d["key"] in wanted]
        if not selected:
            sys.exit(f"未匹配到任何数据集，可用：{[d['key'] for d in config.ALL_DATASETS]}")
    else:
        selected = config.ALL_DATASETS

    all_detail = []
    for ds in selected:
        all_detail.append(run_one_dataset(ds["key"], hri_subjects, n_perm))

    # 汇总：合并本次明细与 results 中已有的其他数据集明细，保证 summary 覆盖全部
    summary = pd.concat(all_detail, ignore_index=True)
    summary.to_csv(config.RESULTS_DIR / "summary_all.csv", index=False)

    # 控制台打印关键结论
    print("\n" + "=" * 64)
    print("全部管线 balanced accuracy 汇总（%）")
    print("=" * 64)
    piv = summary.pivot_table(index=["dataset", "feature"],
                              columns="classifier", values="bacc") * 100
    print(piv.round(2).to_string())
    print("\n结果已保存到：", config.RESULTS_DIR)


if __name__ == "__main__":
    main()
