# -*- coding: utf-8 -*-
"""
evaluation.py
=============
评估指标与交叉验证编排，保证特征提取只在训练折拟合（无数据泄漏）。

内容：
    1. evaluate_pipeline()  : 分层 K 折交叉验证，返回每折汇总的预测与指标
    2. balanced_accuracy()  : 主指标（适合类别不平衡）
    3. tpr_fpr()            : 真阳性率/假阳性率
    4. permutation_test()   : 置换检验，估计 chance 水平与显著性 p
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

import config
from classifiers import build_classifier
from features import build_feature


# ============================================================
# 指标函数
# ============================================================
def balanced_accuracy(y_true, y_pred):
    """
    平衡准确率 = 各类别召回率的算术平均。

    为什么用它：ErrP 数据里错误试次常只占 20% 左右，若全猜“正确”，
    普通准确率也有 80%，但完全没检测到错误。balanced accuracy 会给 50%，
    如实反映“两类都要分得开”。
    """
    return balanced_accuracy_score(y_true, y_pred)


def tpr_fpr(y_true, y_pred):
    """
    返回真阳性率（TPR，错误被正确识别的比例）与
    假阳性率（FPR，正确被误报为错误的比例）。
    """
    cm = confusion_matrix(y_true, y_pred,
                          labels=[config.LABEL_CORRECT, config.LABEL_ERROR])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return float(tpr), float(fpr), cm


def permutation_test(y_true, y_pred, n_perm=1000, seed=config.RANDOM_SEED):
    """
    置换检验：在“预测与标签无真实关系”的零假设下估计 chance 分布。

    做法：保持预测不变，反复随机打乱标签，计算每次的 balanced accuracy，
    得到 null 分布。
        - chance 阈值取 null 的 95% 分位（论文用 95 分位定 chance）；
        - p 值 = (1 + 超过/等于观测值的置换数) / (n_perm + 1)。

    口径说明：这是对“最终预测与标签对齐”的标签置换检验，
    计算高效且是 BCI 文献常用的 chance 估计方式。
    """
    rng = np.random.default_rng(seed)
    observed = balanced_accuracy(y_true, y_pred)
    null = np.empty(n_perm)
    y_true = np.asarray(y_true)
    for i in range(n_perm):
        perm_idx = rng.permutation(len(y_true))
        null[i] = balanced_accuracy(y_true[perm_idx], y_pred)
    chance95 = float(np.quantile(null, 0.95))
    p_value = (1 + np.sum(null >= observed)) / (n_perm + 1)
    return {
        "observed_bacc": float(observed),
        "chance_level_95": chance95,
        "p_value": float(p_value),
        "null_mean": float(np.mean(null)),
    }


# ============================================================
# 交叉验证编排：特征 × 单个分类器
# ============================================================
def evaluate_pipeline(epoch_data, feature_key, classifier_key,
                      n_folds=config.CV_FOLDS, seed=config.RANDOM_SEED,
                      n_perm=config.N_PERMUTATIONS, dataset_key: str = ""):
    """
    用分层 K 折交叉验证评估“一条特征+分类器管线”。

    关键防泄漏点：
        特征提取器（xDAWN/FCB）只在每折的训练数据上 fit，
        再用学到的滤波器 transform 验证折；分类器也只在特征化的训练折 fit。

    返回：
        包含真实/预测标签、bacc、TPR/FPR、混淆矩阵与置换检验的结果字典。
    """
    X, y = epoch_data.X, epoch_data.y
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # 收集每个验证折的真实标签与预测
    oof_true = np.empty_like(y)
    oof_pred = np.empty_like(y)

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # 1) 特征：训练折拟合，再变换训练/验证折
        feat = build_feature(feature_key, epoch_data.ch_names,
                             epoch_data.sfreq, dataset_key=dataset_key)
        feat.fit(X_tr, y_tr)
        Z_tr = feat.transform(X_tr)
        Z_te = feat.transform(X_te)

        # 2) 分类器：只在特征化训练折拟合，对验证折预测
        clf = build_classifier(classifier_key)
        clf.fit(Z_tr, y_tr)
        oof_pred[test_idx] = clf.predict(Z_te)
        oof_true[test_idx] = y_te

    # 汇总指标
    bacc = balanced_accuracy(oof_true, oof_pred)
    tpr, fpr, cm = tpr_fpr(oof_true, oof_pred)
    perm = permutation_test(oof_true, oof_pred, n_perm=n_perm, seed=seed)

    return {
        "feature": feature_key,
        "classifier": classifier_key,
        "bacc": float(bacc),
        "tpr": tpr,
        "fpr": fpr,
        "confusion_matrix": cm.tolist(),
        "permutation": perm,
        "y_true": oof_true,
        "y_pred": oof_pred,
    }
