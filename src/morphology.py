# -*- coding: utf-8 -*-
"""
morphology.py
=============
论文 4.6“ErrP morphology, SNR and discrimination”的复现。

对每个数据集，在指定中线通道（默认 Fz）上提取 ErrP 的形态学指标：
    - 第一负峰（error negativity, ERN）的幅度与潜伏期；
    - 随后正峰（error positivity, Pe）的幅度与潜伏期；
    - 错误/正确两类在选定时间窗内的 template-based SNR（dB）。

所有指标都从真实数据计算，时间轴基于 EpochData.tmin 与 sfreq 推算。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import config
from discriminability import template_snr


# 论文 4.6 逐数据集的 SNR 时间窗（秒，相对事件），顺序对齐论文行 907：
# BNCI,Kaggle,LSC,HRIcursor,HRIrobot,Coadapt,Game,Gaze,TactileV,TactileVT,Virtual
DATASET_SNR_WINDOWS = {
    "LSC": (0.2, 0.7),
    "HRIcursor": (0.0, 0.5),
    "HRIrobot": (0.0, 0.5),
    "Coadaptation": (0.2, 0.7),
    "Gaze": (0.0, 0.5),
    "VirtualCursor": (0.3, 0.8),
    "BNCI013": (0.2, 0.7),
}


def snr_window_for(dataset_key, tmin, sfreq, n_times):
    """
    返回某数据集 SNR 窗在 epoch 内的样本索引（lo, hi）。
    优先用论文特定窗；超出 epoch 边界时裁剪，保证至少有一个点。
    """
    lo_s, hi_s = DATASET_SNR_WINDOWS.get(dataset_key, (0.0, 0.5))
    lo = int(round((lo_s - tmin) * sfreq))
    hi = int(round((hi_s - tmin) * sfreq))
    lo = max(0, min(lo, n_times - 1))
    hi = max(lo + 1, min(hi, n_times))
    return lo, hi


def find_channel(ch_names, target):
    """
    大小写不敏感地查找通道，返回索引；找不到返回 None。
    target 可给多个候选，按顺序尝试。
    """
    targets = [target] if isinstance(target, str) else list(target)
    lower_map = {name.lower(): i for i, name in enumerate(ch_names)}
    for t in targets:
        if t.lower() in lower_map:
            return lower_map[t.lower()]
    return None


def _time_axis(n_times, sfreq, tmin):
    """根据 tmin 与采样率推算每个样本的真实时间（秒）。"""
    return tmin + np.arange(n_times) / sfreq


def to_microvolt(X):
    """
    把数据统一到 µV：MNE 读取的 raw 单位为 V（量级约 1e-3），
    而 .mat 直接加载的数据集本就是 µV（量级 1e1~1e3）。
    依据最大绝对值判定，V 量级时整体乘 1e6，返回转换后的数组与标志。
    """
    X = np.asarray(X, dtype=np.float64)
    if np.abs(X).max() < 0.01:
        return X * 1e6, True
    return X, False


def peak_metrics(wave, times, neg_window=(0.0, 0.3), pos_window=(0.2, 0.6)):
    """
    在单条平均波形上提取负峰与正峰。

    参数
    ----
    wave          : 一维平均波形（µV）
    times         : 对应时间轴（秒）
    neg_window    : 搜索负峰的时间区间（相对事件，秒）
    pos_window    : 搜索正峰的时间区间

    返回 dict：负峰幅度/潜伏期、正峰幅度/潜伏期；区间内无点时对应值为 NaN。
    """
    def _peak(lo, hi, want_min):
        mask = (times >= lo) & (times <= hi)
        if not np.any(mask):
            return np.nan, np.nan
        idx = np.where(mask)[0]
        sub = wave[idx]
        pick = idx[int(np.argmin(sub))] if want_min else idx[int(np.argmax(sub))]
        return float(wave[pick]), float(times[pick])

    neg_amp, neg_lat = _peak(neg_window[0], neg_window[1], True)
    pos_amp, pos_lat = _peak(pos_window[0], pos_window[1], False)
    return {"neg_amp": neg_amp, "neg_latency": neg_lat,
            "pos_amp": pos_amp, "pos_latency": pos_lat}


def dataset_morphology(data, dataset_key=None, channel=("Fz", "Fz"),
                       neg_window=(0.0, 0.3), pos_window=(0.2, 0.6)):
    """
    计算一个数据集在指定通道上的形态学与 SNR 指标。

    返回 dict，包含错误/正确两类的平均波形峰值、潜伏期，以及两类在
    论文选定窗内的 template-based SNR（线性值与 dB）。
    """
    ch_idx = find_channel(data.ch_names, channel)
    if ch_idx is None:
        # 退化：用第一个通道
        ch_idx = 0
    n_times = data.X.shape[2]
    times = _time_axis(n_times, data.sfreq, data.tmin)

    # 统一到 µV（SNR 是比值，单位转换不影响其结果）
    X_uV, _ = to_microvolt(data.X)

    err_mask = data.y == config.LABEL_ERROR
    cor_mask = ~err_mask
    wave_err = np.mean(X_uV[err_mask, ch_idx, :], axis=0)
    wave_cor = np.mean(X_uV[cor_mask, ch_idx, :], axis=0)

    metrics = peak_metrics(wave_err, times, neg_window, pos_window)
    metrics["channel"] = data.ch_names[ch_idx]

    # SNR：在该通道、论文数据集特定时间窗内计算
    if dataset_key:
        t0_idx, t1_idx = snr_window_for(dataset_key, data.tmin,
                                        data.sfreq, n_times)
    else:
        t0_idx = int(round((0.0 - data.tmin) * data.sfreq))
        t1_idx = min(n_times, t0_idx + int(round(0.5 * data.sfreq)))

    snr_err = template_snr(X_uV[err_mask, ch_idx, t0_idx:t1_idx])
    snr_cor = template_snr(X_uV[cor_mask, ch_idx, t0_idx:t1_idx])
    metrics["snr_err_db"] = 10 * np.log10(snr_err) if snr_err > 0 else np.nan
    metrics["snr_cor_db"] = 10 * np.log10(snr_cor) if snr_cor > 0 else np.nan
    return metrics
