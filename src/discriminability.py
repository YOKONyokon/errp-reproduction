# -*- coding: utf-8 -*-
"""
discriminability.py
===================
两类脑电试次“可分性/判别力”的度量工具，公式与官方 MATLAB
（ErrP_APP/HelperFunctions 中的 rsquare.m、gp_var_SNR.m）逐字对齐。

提供两个核心度量：
    1. r_squared()  : 两分布的决定系数 r^2（点二列相关的平方），
                      取值 [0,1]，越大表示该点两类越分得开。
    2. template_snr(): 基于模板的信噪比，模板方差 / 平均噪声方差，
                      对应 Muller 等使用的 template-based SNR。

这两个量在论文中承担三个角色：
    - 4.2 用 r^2 做特征选择；
    - 4.6 用 r^2/SNR 描述判别力与形态；
    - 4.7 把 r^2、SNR 作为自变量，对最佳准确率做线性回归。
"""

import numpy as np


def r_squared(x1, x2):
    """
    计算两批一维样本的 r^2（对齐官方 rsquare.m）。

    数学形式：
        设两类样本和为 S1、S2，样本量 n1、n2，平方和 Q1、Q2，
        合并校正项 G = (S1+S2)^2 / (n1+n2)，则
            r^2 = (S1^2/n1 + S2^2/n2 - G) / (Q1 + Q2 - G)
    分子是“类间差异”，分母是“总差异”，故 r^2 可理解为
    总方差中能被类别解释的比例。

    参数
    ----
    x1, x2 : 一维数组，分别为两类（如错误/正确）在某个点上的取值。

    返回
    ----
    浮点数 r^2（分母为 0 时返回 0，表示该点无判别信息）。
    """
    q = np.asarray(x1, dtype=np.float64)
    r = np.asarray(x2, dtype=np.float64)
    n1, n2 = len(q), len(r)
    if n1 == 0 or n2 == 0:
        return 0.0

    sum1, sum2 = np.sum(q), np.sum(r)
    sumsqu1 = np.sum(q * q)
    sumsqu2 = np.sum(r * r)
    g = (sum1 + sum2) ** 2 / (n1 + n2)

    denom = sumsqu1 + sumsqu2 - g
    if denom <= 1e-12:
        return 0.0
    numer = sum1 ** 2 / n1 + sum2 ** 2 / n2 - g
    return float(numer / denom)


def r2_map(X, y, label_error=1):
    """
    对每个 (通道, 时间点) 计算两类 r^2，得到一张“判别力地形图”。

    参数
    ----
    X : 数组 (n_trials, n_channels, n_times)
    y : 数组 (n_trials,) 类别标签
    label_error : 错误类的标签值（另一类自动视为正确）。

    返回
    ----
    R2 : 数组 (n_channels, n_times)，每个元素为该点的 r^2。
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n_ch, n_t = X.shape[1], X.shape[2]
    err = X[y == label_error]
    cor = X[y != label_error]

    R2 = np.zeros((n_ch, n_t))
    for ch in range(n_ch):
        for t in range(n_t):
            R2[ch, t] = r_squared(err[:, ch, t], cor[:, ch, t])
    return R2


def template_snr(X):
    """
    基于模板的单通道信噪比（对齐官方 gp_var_SNR.m）。

    直觉：把所有试次的平均（grand mean）当作“信号模板”，把每个试次
    减去模板后的残差当作“噪声”。于是
        SNR = Var(模板) / mean(Var(每个试次的残差))
    模板随时间起伏越大、单试次围绕模板越稳定，信噪比越高。

    参数
    ----
    X : 数组 (n_trials, n_times)，单个通道上的多试次信号。

    返回
    ----
    浮点数 SNR（线性比值，非 dB；需要 dB 时用 10*log10 转换）。
    """
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] < 2:
        return 0.0
    template = np.mean(X, axis=0)                 # 信号模板
    residual = X - template[np.newaxis, :]       # 去模板残差
    noise_var = np.mean(np.var(residual, axis=1))
    if noise_var <= 1e-12:
        return 0.0
    return float(np.var(template) / noise_var)


def snr_map(X):
    """
    逐通道计算 template-based SNR。

    参数
    ----
    X : 数组 (n_trials, n_channels, n_times)

    返回
    ----
    SNR : 数组 (n_channels,)，每个通道的线性信噪比。
    """
    X = np.asarray(X, dtype=np.float64)
    n_ch = X.shape[1]
    out = np.empty(n_ch)
    for ch in range(n_ch):
        out[ch] = template_snr(X[:, ch, :])
    return out
