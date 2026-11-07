# -*- coding: utf-8 -*-
"""
splits.py
=========
统一的数据划分协议（阶段 3 固化），并在注释中说明各协议的适用场景与
防泄漏含义。

提供：
    1. within_session_splits()  —— 分层 K 折（试次内，论文主口径）
    2. loso_splits()            —— 留一被试（Leave-One-Subject-Out）
    3. grouped_splits()         —— 通用按组留一（可用于 cross-task）
    4. 泄漏检查：assert_no_overlap / assert_group_disjoint

标签约定：0=正确，1=错误。
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut


def within_session_splits(y, n_folds=5, seed=42):
    """
    分层 K 折（试次内、打乱），返回 [(train_idx, test_idx), ...]。

    口径：在全部试次上按标签分层随机划分，保证每折正负比例与总体一致。
    评估的是"同一人群/会话内对新试次"的分类能力，对齐论文 Yasemin 2023
    的 5-fold stratified CV。注意：同一被试的试次可能跨训练/测试折，
    因此它不是跨被试泛化评估。
    """
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def grouped_splits(groups):
    """
    按组留一（Leave-One-Group-Out），返回 [(train_idx, test_idx), ...]。

    每次把一个组的全部样本作为测试集，其余组作为训练集。
    当 groups=被试标识时即 LOSO；当 groups=任务标识时可用于 cross-task。
    """
    groups = np.asarray(groups)
    logo = LeaveOneGroupOut()
    return list(logo.split(np.zeros(len(groups)), groups=groups))


def loso_splits(subjects):
    """留一被试划分（grouped_splits 的语义化别名）。"""
    return grouped_splits(subjects)


def assert_no_overlap(train_idx, test_idx):
    """训练/测试索引无重叠（样本级防泄漏）。"""
    return set(map(int, train_idx)).isdisjoint(set(map(int, test_idx)))


def assert_group_disjoint(train_idx, test_idx, groups):
    """
    测试集所在的组完全不出现在训练集（组级防泄漏）。

    对 LOSO：测试被试的任何试次都不得进入训练集，
    这是跨被试泛化评估成立的必要条件。
    """
    groups = np.asarray(groups)
    test_groups = set(groups[list(test_idx)].tolist())
    train_groups = set(groups[list(train_idx)].tolist())
    return test_groups.isdisjoint(train_groups)
