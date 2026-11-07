# -*- coding: utf-8 -*-
"""
plotting.py
===========
把复现结果画成可直接查看的图，全部用 matplotlib 保存为 PNG。

包含：
    1. plot_bacc_heatmap()   : 特征×分类器的 balanced accuracy 热力图
    2. plot_confusion()      : 单条管线的混淆矩阵
    3. plot_waveforms()      : 错误 vs 正确的平均 ERP 波形（关键电极）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")          # 无显示环境下用非交互后端
import matplotlib.pyplot as plt
import numpy as np

import config

# 统一配色与字体
plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": "#8aa0bd",
    "axes.labelcolor": "#22324a",
    "axes.titlesize": 13,
    "figure.dpi": 130,
})


def plot_bacc_heatmap(matrix, feature_keys, classifier_keys, title, save_path):
    """
    画 balanced accuracy 热力图（行=特征，列=分类器）。

    matrix : 二维数组，元素为 bacc（0~1）
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0.5, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(classifier_keys)))
    ax.set_xticklabels(classifier_keys)
    ax.set_yticks(range(len(feature_keys)))
    ax.set_yticklabels(feature_keys)
    ax.set_title(title)

    # 每格标注百分比
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]*100:.1f}", ha="center",
                    va="center", color="#10243f", fontweight="bold")
    fig.colorbar(im, ax=ax, label="Balanced accuracy")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm, labels, title, save_path):
    """画混淆矩阵（归一化到比例，便于不平衡数据阅读）。"""
    cm = np.asarray(cm, dtype=float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(2)); ax.set_xticklabels(labels)
    ax.set_yticks(range(2)); ax.set_yticklabels(labels)
    ax.set_xlabel("预测"); ax.set_ylabel("真实")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_norm[i,j]*100:.1f}%\n(n={int(cm[i,j])})",
                    ha="center", va="center", color="#10243f")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_waveforms(epoch_data, save_path, dataset_key):
    """
    画错误 vs 正确的平均 ERP 波形（取中线代表电极）。

    直观验证 ErrP：错误曲线在反馈后应出现额中央负偏转（ERN/Ne）
    与随后的正波（Pe）。
    """
    # 选一个存在的中线电极（大小写不敏感，优先 Cz/FCz/Fz）
    lower_map = {ch.lower(): i for i, ch in enumerate(epoch_data.ch_names)}
    ch_idx = None
    for pref in ("cz", "fcz", "fz"):
        if pref in lower_map:
            ch_idx = lower_map[pref]
            break
    if ch_idx is None:
        ch_idx = 0
    ch = epoch_data.ch_names[ch_idx]

    # 时间轴：从数据自身 tmin 开始，按点数与采样率推算真实结束时刻。
    # 避免把 Gaze 的 [-0.2,0.5]（180 点）错误拉伸到 [-0.2,0.8]。
    n_times = epoch_data.X.shape[2]
    t0 = getattr(epoch_data, "tmin", config.EPOCH_TMIN_S)
    t = t0 + np.arange(n_times) / epoch_data.sfreq

    err = epoch_data.X[epoch_data.y == config.LABEL_ERROR, ch_idx, :].mean(0)
    cor = epoch_data.X[epoch_data.y == config.LABEL_CORRECT, ch_idx, :].mean(0)

    # 单位统一：MNE 读取的连续数据以 V 存储（量级约 1e-6），
    # 检测到小量级时换算为 µV，与 LSC 的 µV 口径保持一致。
    if np.max(np.abs(epoch_data.X[:, ch_idx, :])) < 1e-3:
        err, cor = err * 1e6, cor * 1e6

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(t, err, color="#d6455c", lw=2, label="错误 Error")
    ax.plot(t, cor, color="#2f6fd0", lw=2, label="正确 Correct")
    ax.axvline(0, color="#7a8aa3", ls="--", lw=1)
    ax.axhline(0, color="#c3cede", lw=0.8)
    ax.set_xlabel("时间 (s，相对反馈)")
    ax.set_ylabel("电压 (µV)")
    ax.set_title(f"{dataset_key} · 电极 {ch} 的平均 ERRP 波形")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
