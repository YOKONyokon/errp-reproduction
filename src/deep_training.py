# -*- coding: utf-8 -*-
"""
deep_training.py — 深度模型训练/评估工具

封装 CPU 上训练 EEGNet 的完整流程：
- 折内逐通道标准化（统计量只来自训练折，避免泄漏）；
- balanced 交叉熵（按训练折类别频次给权重，应对类别不平衡）；
- 从训练折再分层切出 20% 验证集做早停，保留验证集最优权重；
- 输出测试折预测与 balanced accuracy。
"""
import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit

from deep_models import EEGNet
from evaluation import balanced_accuracy

torch.set_num_threads(max(1, torch.get_num_threads()))


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _standardize(Xtr, Xte):
    mean = Xtr.mean(axis=(0, 2), keepdims=True)
    std = Xtr.std(axis=(0, 2), keepdims=True)
    std[std == 0] = 1.0
    return (Xtr - mean) / std, (Xte - mean) / std


def train_eegnet(Xtr, ytr, Xte, yte, n_channels, n_times,
                 seed=42, epochs=30, batch_size=128, lr=1e-3,
                 patience=6, verbose=False):
    set_seed(seed)
    Xtr, Xte = _standardize(Xtr, Xte)

    # 训练折内分层切验证集用于早停
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    ti, vi = next(sss.split(Xtr, ytr))
    Xfit, yfit = Xtr[ti], ytr[ti]
    Xval, yval = Xtr[vi], ytr[vi]

    def tensor(X):
        return torch.tensor(X[:, None, :, :], dtype=torch.float32)

    Xfit_t, Xval_t, Xte_t = tensor(Xfit), tensor(Xval), tensor(Xte)
    yfit_t = torch.tensor(yfit, dtype=torch.long)

    # balanced 类别权重
    counts = np.bincount(yfit, minlength=2)
    weights = len(yfit) / (2 * np.maximum(counts, 1))
    weight_t = torch.tensor(weights, dtype=torch.float32)

    model = EEGNet(n_channels, n_times)
    criterion = nn.CrossEntropyLoss(weight=weight_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_state, best_val, wait = None, -np.inf, 0
    n_batches = int(np.ceil(len(yfit) / batch_size))
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(yfit))
        for b in range(n_batches):
            bidx = perm[b * batch_size:(b + 1) * batch_size]
            optimizer.zero_grad()
            loss = criterion(model(Xfit_t[bidx]), yfit_t[bidx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xval_t).argmax(dim=1).numpy()
        val_bacc = balanced_accuracy(yval, val_pred)
        if verbose:
            print(f"epoch {ep:3d} val_bacc={val_bacc:.3f}")
        if val_bacc > best_val:
            best_val, wait = val_bacc, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(Xte_t).argmax(dim=1).numpy()
    test_bacc = balanced_accuracy(yte, test_pred)
    return test_pred, test_bacc
