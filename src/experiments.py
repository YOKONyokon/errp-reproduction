# -*- coding: utf-8 -*-
"""
experiments.py
==============
论文第 4 节“受控对比实验”（4.2–4.5）的 Python 复现。所有处理都在
交叉验证的训练折内拟合，再应用到验证折，杜绝数据泄漏。

四个实验（每个都是“无处理 vs 有处理”的配对比较，主分类器用 sLDA）：
    4.2 Feature selection  : 无选择 / r^2 选择 / PCA
    4.3 Time window        : 全时间窗 / 按判别力选择的时间窗
    4.4 Augmentation       : 单窗预测 / 双时间窗 AND 决策
    4.5 Downsampling       : 原始采样 / 窗口平均降采样

口径对齐说明：
    - r^2 选择与 SNR 见 discriminability.py（公式对齐官方 MATLAB）；
    - 增强对应论文 3.2.4：同一事件取两个时间定位的窗，两个预测都正确
      才判正确（AND），论文明确它“不增加训练数据量”；
    - 因本项目 epoch 已固定切好，偏移窗在 epoch 内部用真实数据取，
      不做任何填充，保证结果完全可追溯。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA

import config
from discriminability import r_squared
from features import build_feature
from classifiers import build_classifier


# ============================================================
# 4.2 特征选择（作用于特征提取后的二维矩阵）
# ============================================================
def r2_feature_indices(Z_tr, y_tr, n_keep):
    """
    在训练折上逐特征列计算 r^2，返回判别力最强的 n_keep 个列索引。

    返回的索引按 r^2 降序排列；若特征总数不足 n_keep，则全部保留。
    """
    n_feat = Z_tr.shape[1]
    n_keep = min(n_keep, n_feat)
    scores = np.empty(n_feat)
    for j in range(n_feat):
        scores[j] = r_squared(Z_tr[y_tr == config.LABEL_ERROR, j],
                              Z_tr[y_tr != config.LABEL_ERROR, j])
    order = np.argsort(-scores)
    return np.sort(order[:n_keep])          # 排序后保持时间/特征原顺序


def pca_reduce(Z_tr, Z_te, n_components):
    """
    用 PCA 降维：仅在训练折拟合，再同时变换训练/验证折。
    返回降维后的两个矩阵（成分数不超过样本数与特征数）。
    """
    k = min(n_components, Z_tr.shape[1], Z_tr.shape[0])
    pca = PCA(n_components=k)
    return pca.fit_transform(Z_tr), pca.transform(Z_te)


# ============================================================
# 4.3 时间窗选择（在 epoch 时间维度上，按 r^2 判别力）
# ============================================================
def _r2_time_curve(X_tr, y_tr):
    """逐时间点计算跨通道平均 r^2，返回判别力时间曲线。"""
    n_ch, n_t = X_tr.shape[1], X_tr.shape[2]
    curve = np.zeros(n_t)
    for t in range(n_t):
        s = 0.0
        for ch in range(n_ch):
            s += r_squared(X_tr[y_tr == config.LABEL_ERROR, ch, t],
                           X_tr[y_tr != config.LABEL_ERROR, ch, t])
        curve[t] = s / n_ch
    return curve


def choose_time_window(X_tr, y_tr, sfreq, target_s=0.5, min_s=0.2):
    """
    基于训练折的 r^2 时间曲线选择判别时间窗（对齐论文 3.2.3/4.6 的
    AUC 方法：保留一个约 500ms、覆盖主要判别区间的连续窗）。

    做法：逐时间点计算跨通道平均 r^2，再用固定长度的滑动窗搜索
    “窗内 r^2 总和最大”的连续区间，保证窗口足够长、不会因只取峰值
    附近的短区间而丢失信息。目标长度 500ms，epoch 较短时退化为
    不短于 200ms；若整个 epoch 都不足目标长度，则返回全部时间点。
    """
    n_t = X_tr.shape[2]
    curve = _r2_time_curve(X_tr, y_tr)

    target = int(round(target_s * sfreq))
    win_len = min(n_t, max(target, 1))
    if win_len >= n_t:
        return np.arange(n_t)

    min_len = int(round(min_s * sfreq))
    win_len = max(win_len, min(min_len, n_t))

    # 滑动窗找累计判别力最大的区间
    best_start, best_sum = 0, -np.inf
    cumsum = np.concatenate(([0.0], np.cumsum(curve)))
    for start in range(0, n_t - win_len + 1):
        window_sum = cumsum[start + win_len] - cumsum[start]
        if window_sum > best_sum:
            best_sum, best_start = window_sum, start
    return np.arange(best_start, best_start + win_len)


# ============================================================
# 4.5 降采样（窗口平均，对应官方 dsmethod=2）
# ============================================================
def downsample_factor_for_sfreq(sfreq):
    """
    返回各采样率下的降采样因子（对齐官方脚本 fs/2、fs/4 的取法）。
    256 Hz 取因子 4（降到 64 Hz），512 Hz 取因子 8（降到 64 Hz）。
    """
    return 8 if sfreq >= 500 else 4


def downsample_epoch(X, factor):
    """
    用不重叠窗口平均做降采样（训练/验证折用同一规则，无拟合参数）。
    X: (n, ch, t) -> (n, ch, floor(t/factor))
    """
    n, ch, t = X.shape
    t2 = (t // factor) * factor
    Xc = X[:, :, :t2].reshape(n, ch, t2 // factor, factor)
    return Xc.mean(axis=3)


# ============================================================
# 单折内：特征化 + 可选处理，返回训练/验证特征
# ============================================================
def _featurize_with_options(data, X_tr, y_tr, X_te, feat_key,
                            fs_method, n_keep, pca_comp):
    """
    在给定训练/验证 epoch 上做特征提取，并按 4.2/4.3/4.5 的要求处理。

    fs_method : 'none' / 'r2' / 'pca'（特征选择方式）
    返回 (Z_tr, Z_te) 二维特征矩阵。
    """
    feat = build_feature(feat_key, data.ch_names, data.sfreq)
    feat.fit(X_tr, y_tr)
    Z_tr = feat.transform(X_tr)
    Z_te = feat.transform(X_te)

    if fs_method == 'r2':
        cols = r2_feature_indices(Z_tr, y_tr, n_keep)
        Z_tr, Z_te = Z_tr[:, cols], Z_te[:, cols]
    elif fs_method == 'pca':
        Z_tr, Z_te = pca_reduce(Z_tr, Z_te, pca_comp)
    return Z_tr, Z_te


def _bacc(y_true, y_pred):
    """两类召回率平均（balanced accuracy）。"""
    from sklearn.metrics import balanced_accuracy_score
    return balanced_accuracy_score(y_true, y_pred)


# ============================================================
# 统一交叉验证：评估某一实验条件下的一条特征×分类器管线
# ============================================================
def run_condition(data, feat_key, clf_key, experiment,
                  n_folds=config.CV_FOLDS, seed=config.RANDOM_SEED,
                  n_keep=200, pca_comp=2):
    """
    运行一个实验条件，返回 out-of-fold balanced accuracy。

    experiment 取值与对应处理：
        'baseline'   : 基线（无额外处理）
        'fs_r2'      : 4.2 r^2 特征选择
        'fs_pca'     : 4.2 PCA 特征选择
        'window'     : 4.3 时间窗选择
        'downsample' : 4.5 窗口平均降采样
        'augment'    : 4.4 双时间窗 AND 决策

    说明：增强在决策层实现，其余处理在训练折内拟合。
    """
    X, y = data.X, data.y
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_true = np.empty_like(y)
    oof_pred = np.empty_like(y)

    factor = downsample_factor_for_sfreq(data.sfreq)

    for tr, te in skf.split(X, y):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        # ---- 4.4 增强：独立走“双等长错位窗 + AND”路径 ----
        if experiment == 'augment':
            oof_pred[te] = _augment_and_predict(
                data, X_tr, X_te, y_tr, feat_key, clf_key)
            oof_true[te] = y_te
            continue

        # ---- 4.5 降采样（数据层，规则固定，先于特征提取）----
        if experiment == 'downsample':
            X_tr = downsample_epoch(X_tr, factor)
            X_te = downsample_epoch(X_te, factor)
            ds_sfreq = data.sfreq / factor
        else:
            ds_sfreq = data.sfreq

        # ---- 4.3 时间窗选择（训练折确定索引，应用到两侧）----
        if experiment == 'window':
            t_idx = choose_time_window(X_tr, y_tr, data.sfreq)
            X_tr = X_tr[:, :, t_idx]
            X_te = X_te[:, :, t_idx]

        # ---- 特征提取 + 4.2 特征选择 ----
        fs_method = 'none'
        if experiment == 'fs_r2':
            fs_method = 'r2'
        elif experiment == 'fs_pca':
            fs_method = 'pca'
        Z_tr, Z_te = _featurize_with_options(
            _wrap(data, ds_sfreq), X_tr, y_tr, X_te, feat_key,
            fs_method, n_keep, pca_comp)

        # ---- 分类 ----
        clf = build_classifier(clf_key)
        clf.fit(Z_tr, y_tr)
        oof_pred[te] = clf.predict(Z_te)
        oof_true[te] = y_te

    return float(_bacc(oof_true, oof_pred))


def _augment_and_predict(data, X_tr, X_te, y_tr, feat_key, clf_key):
    """
    4.4 增强的决策层实现。

    在已切好的 epoch 内，用两个等长的错位子窗近似论文的两种切分：
        窗 A（正常锁定，取后段）：[:, :, shift:T]
        窗 B（前移 100ms，取前段）：[:, :, 0:T-shift]
    两者长度均为 T-shift，特征维度一致；各自独立训练分类器并预测，
    最终仅当两个预测都为正确(0)才判正确，否则判错误(1)。
    """
    T = X_tr.shape[2]
    shift = int(round(0.1 * data.sfreq))
    if 2 * shift >= T:
        shift = max(1, T // 4)

    # 两个等长错位窗
    X_tr_a, X_te_a = X_tr[:, :, shift:], X_te[:, :, shift:]
    X_tr_b, X_te_b = X_tr[:, :, :T - shift], X_te[:, :, :T - shift]

    # 各自特征化（不做额外特征选择）并独立训练
    Z_tr_a, Z_te_a = _featurize_with_options(
        data, X_tr_a, y_tr, X_te_a, feat_key, 'none', 200, 2)
    Z_tr_b, Z_te_b = _featurize_with_options(
        data, X_tr_b, y_tr, X_te_b, feat_key, 'none', 200, 2)

    clf_a = build_classifier(clf_key)
    clf_a.fit(Z_tr_a, y_tr)
    pred_a = clf_a.predict(Z_te_a)

    clf_b = build_classifier(clf_key)
    clf_b.fit(Z_tr_b, y_tr)
    pred_b = clf_b.predict(Z_te_b)

    # AND：两个预测都为正确才判正确，否则错误
    return np.where(
        (pred_a == config.LABEL_CORRECT) & (pred_b == config.LABEL_CORRECT),
        config.LABEL_CORRECT, config.LABEL_ERROR)


def _wrap(data, sfreq):
    """构造一个轻量对象，暴露特征提取所需的 ch_names/sfreq。"""
    class _D:
        pass
    d = _D()
    d.ch_names = data.ch_names
    d.sfreq = sfreq
    return d
