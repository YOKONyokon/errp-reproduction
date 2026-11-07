# -*- coding: utf-8 -*-
"""
channel_mapping.py — 跨数据集通道映射

为跨任务迁移提供统一通道口径：
- ``HOMOLOG_27``：三个 27 通道同构数据集（HRIcursor/HRIrobot/Coadaptation）
  的完整 10-20 布局，跨这三者迁移时信息零损失。
- ``COMMON_4``：全部 6 个数据集都包含的 4 个共享电极（C3/C4/Cz/Fz），
  用于覆盖全部数据集的迁移，信息损失较大。

通道名在匹配时统一大写、去空格，以容忍不同来源的命名差异。
"""
import numpy as np

from data_loader import EpochData

# 6 集全部共享的 4 电极
COMMON_4 = ["C3", "C4", "Cz", "Fz"]

# 三个 27ch 同构数据集的完整通道（标准 10-20）
HOMOLOG_27 = [
    "Fp1", "Fz", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1",
    "Pz", "P3", "P7", "O1", "O2", "P4", "P8", "CP6", "CP2", "Cz",
    "C4", "T8", "FC6", "FC2", "F4", "F8", "Fp2",
]

# 同构三集的 key，便于按口径批量选择
HOMOLOG_KEYS = ["HRIcursor", "HRIrobot", "Coadaptation"]


def _norm(name: str) -> str:
    return name.strip().upper()


def select_channels(data: EpochData, target_channels) -> EpochData:
    """按 ``target_channels`` 顺序从 EpochData 选取通道，返回新 EpochData。

    若目标通道在数据中不存在，抛出 ValueError 并列出缺失项，避免静默丢通道。
    """
    upper = {_norm(c): i for i, c in enumerate(data.ch_names)}
    idx, missing = [], []
    for ch in target_channels:
        if _norm(ch) in upper:
            idx.append(upper[_norm(ch)])
        else:
            missing.append(ch)
    if missing:
        raise ValueError(f"通道缺失 {missing}；可用 {data.ch_names}")
    idx = np.array(idx)
    return EpochData(
        X=data.X[:, idx, :],
        y=data.y,
        ch_names=[data.ch_names[i] for i in idx],
        sfreq=data.sfreq,
        subjects=data.subjects,
    )


def common_channels(datas: dict) -> list:
    """计算 {name: EpochData} 的通道交集（标准化命名，排序返回）。"""
    sets = [set(_norm(c) for c in d.ch_names) for d in datas.values()]
    return sorted(set.intersection(*sets))
