# -*- coding: utf-8 -*-
"""
deep_models.py — 轻量深度模型（CPU 友好）

标准 EEGNet（Lawhern et al. 2018, J. Neural Eng.）的独立 PyTorch 实现，
不依赖 ERP-Benchmark 的 configs 框架。输入张量形状 (batch, 1, channels, times)。
为适配 4 核 CPU，默认通道数取较小值（F1=8、D=2、F2=16）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    def __init__(self, n_channels, n_times, n_classes=2,
                 F1=8, D=2, F2=16, kernel_length=63, dropout=0.5):
        super().__init__()
        # ---- Block 1：时间卷积 + 深度空间卷积 ----
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length),
                               padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthconv = nn.Conv2d(F1, F1 * D, (n_channels, 1),
                                   groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # ---- Block 2：可分离卷积 ----
        self.sepconv = nn.Conv2d(F1 * D, F1 * D, (1, 15),
                                 padding=(0, 7), groups=F1 * D, bias=False)
        self.pointwise = nn.Conv2d(F1 * D, F2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        self.fc = nn.Linear(self._feature_size(n_channels, n_times), n_classes)

    def _feature_size(self, C, T):
        with torch.no_grad():
            out = self.forward_features(torch.zeros(1, 1, C, T))
            return out.reshape(1, -1).shape[1]

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthconv(x)
        x = F.elu(self.bn2(x))
        x = self.drop1(self.pool1(x))

        x = self.sepconv(x)
        x = self.pointwise(x)
        x = F.elu(self.bn3(x))
        x = self.drop2(self.pool2(x))
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)


def build_deep_model(key, n_channels, n_times, n_classes=2):
    """模型工厂，便于后续扩展其他网络。"""
    if key == "EEGNet":
        return EEGNet(n_channels, n_times, n_classes)
    raise ValueError(f"未知深度模型 {key}")
