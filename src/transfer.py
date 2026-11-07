# -*- coding: utf-8 -*-
"""
transfer.py — 跨任务迁移的子空间/域对齐工具

实现欧氏对齐（Euclidean Alignment, Zanini et al. 2018）：
对每个数据集，用其试次协方差的均值 R 作为参考，把每个试次左乘 R^{-1/2}，
使该数据集的平均协方差变为单位阵。不同数据集各自对齐后，二阶统计量被拉到
同一子空间，从而减小源域/目标域的分布偏移，是 BCI 迁移学习的经典子空间正则。
"""
import numpy as np
from scipy.linalg import fractional_matrix_power


def trial_covariances(X):
    """X (n, C, T) -> 逐试次样本协方差 (n, C, C)。"""
    n, C, T = X.shape
    covs = np.empty((n, C, C))
    for i in range(n):
        x = X[i]
        x = x - x.mean(axis=1, keepdims=True)
        denom = max(T - 1, 1)
        covs[i] = (x @ x.T) / denom
    return covs


def reference_matrix(covs, shrink_frac=1e-6):
    """参考协方差 R = 试次协方差均值，加相对量级的对角项保证正定。

    shrink 取平均对角能量（trace/C）的一个小比例，而非固定绝对值，
    以兼容 V（协方差~1e-10）与 µV（协方差~1e2）两种单位。
    """
    R = covs.mean(axis=0)
    C = R.shape[0]
    scale = np.trace(R) / C
    R = R + np.eye(C) * scale * shrink_frac
    return R


def euclidean_align(X, ref=None):
    """对齐 X；返回 (aligned_X, ref)。源/目标域各自调用得到各自的 ref。"""
    covs = trial_covariances(X)
    if ref is None:
        ref = reference_matrix(covs)
    R_inv_sqrt = fractional_matrix_power(ref, -0.5)
    R_inv_sqrt = np.real_if_close(R_inv_sqrt).astype(np.float64)
    aligned = np.array([R_inv_sqrt @ X[i] for i in range(len(X))])
    return aligned, ref
