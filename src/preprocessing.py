# -*- coding: utf-8 -*-
"""
preprocessing.py
================
EEG 预处理工具函数，并在注释中解释每一步的数学原理与作用。

包含：
    1. butter_bandpass()       —— Butterworth 带通滤波器（零相位 filtfilt）
    2. baseline_correct()      —— 基线校正
    3. common_average_reference() —— 共平均参考（CAR，可选）

说明：
    - HRI 连续数据的主流程直接调用 MNE 官方的 ``raw.filter()``（更稳健、
      自动处理边界），本模块的 scipy 版本用于已切分的 epoch 数据或原理对照；
    - 论文 Yasemin 2023 统一做 1–10 Hz 带通，未统一做 CAR。
"""

import numpy as np
from scipy.signal import butter, filtfilt


# ============================================================
# 1. Butterworth 带通滤波
# ============================================================
def butter_bandpass_coeffs(low_hz: float, high_hz: float, sfreq: float,
                           order: int = 4):
    """
    设计一个 Butterworth（巴特沃斯）带通滤波器的系数。

    数学直觉：
        Butterworth 滤波器的幅频响应在通带内“最平坦”，像一个平滑的斜坡，
        把通带外的低频漂移（如呼吸、缓慢运动）和高频噪声（肌电、工频）压下去，
        同时尽量保留 1–10 Hz 内 ErrP 的 ERN/Pe 波形。

    参数：
        low_hz, high_hz : 通带边界（Hz）
        sfreq           : 采样率（Hz）
        order           : 滤波器阶数，越大截止越陡，但越易产生边界伪影
    返回：
        b, a : 滤波器的分子、分母系数（差分方程系数）
    """
    # Nyquist（奈奎斯特）频率 = 采样率的一半，是可被表示的最高频率
    nyq = sfreq / 2.0
    # 归一化频率（scipy 要求以 0~1 表示，1 对应 Nyquist）
    low_norm = low_hz / nyq
    high_norm = high_hz / nyq
    b, a = butter(order, [low_norm, high_norm], btype="bandpass", output="ba")
    return b, a


def butter_bandpass(data: np.ndarray, low_hz: float, high_hz: float,
                    sfreq: float, order: int = 4, axis: int = -1) -> np.ndarray:
    """
    对信号做零相位（zero-phase）带通滤波。

    为什么用 filtfilt：
        普通滤波会让波形在时间上“滞后”；filtfilt 正向滤波一次、再反向滤波一次，
        相位偏移互相抵消（零相位），因此 ERN 的潜伏期不会被人为移动，
        这对锁定到反馈时刻的 ErrP 分析至关重要。

    参数：
        data : 一维时间序列，或含时间轴的多维数组
        axis : 时间所在的轴
    """
    b, a = butter_bandpass_coeffs(low_hz, high_hz, sfreq, order)
    return filtfilt(b, a, data, axis=axis)


# ============================================================
# 2. 基线校正
# ============================================================
def baseline_correct(epochs: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """
    基线校正：减去反馈前安静段的平均电压。

    公式：x_corrected(t) = x(t) - mean_{t in baseline} x(t)

    作用：不同电极/试次可能带有不同的直流偏置，减去基线后，
    大家都从“同一零点”出发，错误与正确的相对差异才可比。

    参数：
        epochs   : (n_trials, n_channels, n_times)
        baseline : 基线区间对应的时间点索引数组
    """
    baseline_mean = np.mean(epochs[:, :, baseline], axis=2, keepdims=True)
    return epochs - baseline_mean


# ============================================================
# 3. 共平均参考（CAR，可选；论文未统一使用）
# ============================================================
def common_average_reference(data: np.ndarray) -> np.ndarray:
    """
    共平均参考（Common Average Reference, CAR）。

    公式（对每个时间点、每个电极 i）：
        V_i^CAR = V_i - (1/N) * sum_{j=1..N} V_j

    直觉：所有电极共有的“全脑同涨同落”成分（如远处共模噪声）被平均后减掉，
    留下各电极相对于全脑平均的局部活动。属于一种空间高通滤波。

    参数：
        data : (..., n_channels, n_times)，通道轴为倒数第 2 维
    """
    avg = np.mean(data, axis=-2, keepdims=True)   # 跨电极平均
    return data - avg
