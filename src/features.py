# -*- coding: utf-8 -*-
"""
features.py
===========
五种特征提取方法，统一实现为 sklearn 风格的 ``fit()`` / ``transform()`` 接口，
以便在交叉验证的“训练折”内拟合、在“验证折”上变换，杜绝数据泄漏。

五种特征（对应论文 Yasemin 2023 第 3.2.5 节 Feature extraction）：
    1. FCzCzFeature     : 仅用中线电极（FCz/Cz；LSC 无 FCz 时退化为 Fz/Cz）
    2. TimepointsFeature: 全电极时间点，不做空间变换（直接展平）
    3. OverlapFeature   : 重叠窗口平均（100 ms 窗、50 ms 步长）
    4. XDawnFeature     : 直接调用 MNE 官方 ``mne.preprocessing.Xdawn``
    5. FCBFeature       : 直接调用 FCB 官方 Python 工具箱（gpiresML，MIT）

数学原理速览：
    - xDAWN：用广义特征值分解找空间滤波器，使“类信号 vs 类内噪声”的信噪比最大；
    - FCB：在 Fisher 准则（类间离散度 / 类内离散度）下求广义特征向量，
      输出按可分性排序的空间滤波成分。
"""

import sys
from pathlib import Path

# 保证能找到项目根 config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mne
import numpy as np
from scipy import linalg as sla
from scipy.sparse import csr_matrix
from mne.preprocessing import Xdawn

import config

# 载入官方 FCB 工具箱（把其 FCB_toolbox 目录加入路径）
sys.path.append(str(config.FCB_TOOLBOX_DIR / "FCB_toolbox"))
import FCB_toolbox  # 官方实现：FCB_spatial_filters / FCB_projections


# ============================================================
# 特征基类
# ============================================================
class BaseFeature:
    """统一接口基类；子类需实现 _fit 与 _apply。"""

    def fit(self, X: np.ndarray, y: np.ndarray):
        """在训练数据上学习滤波器/选择规则。"""
        self._fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """把 (n,ch,t) 数据变换为二维特征矩阵 (n, n_features)。"""
        return self._apply(X)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)


# ============================================================
# 特征 1：中线电极 FCz/Cz（基线）
# ============================================================
class FCzCzFeature(BaseFeature):
    """
    只取中线电极的时间信号并展平，作为“不做空间滤波”的基线。

    选择逻辑（兼容不同数据集的通道差异）：
        优先 [FCz, Cz]；若数据集没有 FCz（如 LSC），退而用 [Fz, Cz]。
    """

    # 候选中线电极，按优先级匹配
    CANDIDATES = ["FCz", "Cz", "Fz"]

    def _fit(self, X, y):
        ch_names = self.ch_names
        # 建立大小写不敏感映射：兼容 Gaze 的全大写（FCZ/CZ）与
        # 其他数据集的标准写法（FCz/Cz）
        lower_map = {ch.lower(): i for i, ch in enumerate(ch_names)}
        picked_idx = []
        picked_names = []
        for cand in self.CANDIDATES:
            if cand.lower() in lower_map:
                idx = lower_map[cand.lower()]
                picked_idx.append(idx)
                picked_names.append(ch_names[idx])
        # 保证至少取到 2 个，不足则取全部匹配
        self.picked_names = picked_names[:2] if len(picked_idx) >= 2 else picked_names
        self.picked_idx = picked_idx[:2] if len(picked_idx) >= 2 else picked_idx

    def _apply(self, X):
        # 选取通道后展平：(n, n_picked, t) -> (n, n_picked*t)
        sub = X[:, self.picked_idx, :]
        return sub.reshape(len(sub), -1)

    def __init__(self, ch_names):
        self.ch_names = ch_names


# ============================================================
# 特征 2：xDAWN（MNE 官方实现）
# ============================================================
# ------------------------------------------------------------
# 空间滤波共用辅助函数
# ------------------------------------------------------------
def _sparse_design_matrix(y, n_trials, t):
    """
    构造官方 xDAWN 的稀疏 Toeplitz 设计矩阵 D（t*n_trials, t）。

    每个 target（错误）事件在其起始位置放置一个长度为 t 的“单位脉冲序列”，
    用于最小二乘估计错误类的时间原型。用 CSR 稀疏存储，避免 5000+ 试次时
    稠密矩阵占用数 GB 内存（对应 MATLAB 的 sparse D）。
    """
    target_starts = np.where(y == config.LABEL_ERROR)[0] * t
    rows, cols = [], []
    for i in range(t):
        r = target_starts + i
        msk = (r >= 0) & (r < t * n_trials)
        rows.append(r[msk])
        cols.append(np.full(msk.sum(), i))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    return csr_matrix((np.ones(len(rows)), (rows, cols)),
                      shape=(t * n_trials, t))


def _per_trial_zscore(Z):
    """
    逐试次跨特征 z-score，对应官方 normalize2dData.m：
    对每个试次（行）在其全部特征上减均值、除标准差。
    与逐特征 StandardScaler 不同，这里消除的是单个试次整体幅度/偏移的影响。
    """
    mu = Z.mean(axis=1, keepdims=True)
    sd = Z.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (Z - mu) / sd


class XDawnFeature(BaseFeature):
    """
    忠实复刻官方 ErrP_APP 的 xDAWN（xdawn_spatial_filters.m 的广义特征值
    路径），而非 MNE 的 Xdawn 包装。

    xDAWN 直觉：错误反馈会在脑电上叠加一个固定的“类原型”响应，但它淹没在
    噪声里。xDAWN 学一组空间滤波器，使“原型信号能量 / 全信号能量”最大，
    类似给每个电极找最佳加权组合，把错误波形对齐放大。

    与 MNE 版本的关键差异（经 LSC/HRIcursor/Gaze 实测，官方口径显著更接近
    论文 Table 3）：
      1. 只估计并保留 target（错误）类方向的成分，MNE 对两类各取 n_components；
      2. 用广义特征值（原型协方差 vs 全信号协方差），不做 ledoit_wolf 收缩；
      3. 成分数逐数据集取 config.SPATIAL_COMPONENTS（2 或 4）；
      4. 输出做逐试次 z-score（官方 normalize=1）。
    """

    def __init__(self, ch_names, sfreq, n_components: int = None,
                 normalize: bool = True):
        self.ch_names = ch_names
        self.sfreq = sfreq
        self.n_components = n_components
        self.normalize = normalize

    def _fit(self, X, y):
        n, k, t = X.shape
        if self.n_components is None:
            self.n_components = min(k, config.SPATIAL_COMPONENTS.get(
                getattr(self, "_dataset", ""), 2))
        # 拼接全部试次：(t*n, k)，与 MATLAB reshape 一致
        data = np.ascontiguousarray(
            np.transpose(X, (0, 2, 1)).reshape(t * n, k), dtype=np.float64)
        D = _sparse_design_matrix(y, n, t)

        # 小矩阵：D'D (t,t)、D'X (t,k)、X'X (k,k)
        DtD = (D.T @ D).toarray()
        DtX = D.T @ data
        XtX = data.T @ data
        # 原型 A = (D'D)\(D'X)，(t, k)
        A = sla.solve(DtD, DtX, assume_a="pos")
        # 广义特征值：(A' D'D A) w = λ (X'X) w
        numer = A.T @ DtD @ A
        numer = (numer + numer.T) / 2
        XtX = (XtX + XtX.T) / 2
        _, evecs = sla.eigh(numer, XtX)
        # eigh 按特征值升序返回，取末尾 n_components 个（最大者），并翻转成降序
        self.W = evecs[:, -self.n_components:][:, ::-1]

    def _apply(self, X):
        n, k, t = X.shape
        data = np.ascontiguousarray(
            np.transpose(X, (0, 2, 1)).reshape(t * n, k), dtype=np.float64)
        filt = data @ self.W                  # (t*n, n_components)
        # 还原成 (n, n_components, t) 后展平
        back = filt.reshape(n, t, -1).transpose(0, 2, 1)
        Z = back.reshape(n, -1)
        return _per_trial_zscore(Z) if self.normalize else Z


# ============================================================
# 特征 3：FCB（官方 FCB_toolbox，MIT）
# ============================================================
class FCBFeature(BaseFeature):
    """
    直接调用官方 FCB 工具箱的 Fisher Criterion Beamformer。

    Fisher 准则：J(W) = (类间离散度) / (类内离散度)，
    最大化 J 的空间滤波器让两类“中心尽量分开、各自尽量聚拢”。
    官方实现里对类内协方差做正则化（th 参数）并用广义特征值求解。
    """

    def __init__(self, n_components: int = 3, reg_th: float = 0.1,
                 normalize: bool = True):
        self.n_components = n_components
        self.reg_th = reg_th          # 官方示例默认 0.1
        self.normalize = normalize   # 官方 fcb_feature_extraction norm=1

    def _fit(self, X, y):
        # 官方函数要求 (通道×时间×试次)，按类别拆分
        z_err = np.transpose(X[y == config.LABEL_ERROR], (1, 2, 0))
        z_cor = np.transpose(X[y == config.LABEL_CORRECT], (1, 2, 0))
        # 官方源码：返回特征向量 U（空间滤波器）与特征值 V
        U, V = FCB_toolbox.FCB_spatial_filters(
            z_err, z_cor, self.reg_th)
        # np.linalg.eig 对对称矩阵可能返回“虚部为 0”的复数 dtype；
        # 对称矩阵的特征值理论上为实数，这里取实部以兼容 sklearn。
        self.U = np.real(U)
        self.V = np.real(V)

    def _apply(self, X):
        # 取前 n_components 个最具可分性的滤波器做投影
        U_sel = self.U[:, :self.n_components]
        # 投影：对每个试次左乘 U^T
        proj = np.einsum("cd,ndt->nct", U_sel.T, X)
        Z = proj.reshape(len(proj), -1)
        # 逐试次 z-score（对应官方 normalize2dData）
        return _per_trial_zscore(Z) if self.normalize else Z


# ============================================================
# 特征 4：全电极时间点（Timepoints，无空间变换基线）
# ============================================================
class TimepointsFeature(BaseFeature):
    """
    不做任何空间变换，直接把“所有电极 × 所有时间点”展平成一维特征。

    与 FCzCz 的区别：FCzCz 只取 2 个中线电极，而 Timepoints 保留全部电极。
    论文指出该特征维度很高（电极数 × 时间点数），通常需配合特征选择
    （PCA 或 r-squared）使用；本项目保留其“全电极原始”形态作为对照基线。
    """

    def __init__(self):
        # 无需学习参数，fit 仅为满足统一接口
        pass

    def _fit(self, X, y):
        return

    def _apply(self, X):
        # (n, ch, t) -> (n, ch*t)
        return X.reshape(len(X), -1)


# ============================================================
# 特征 5：重叠窗口平均（Overlapped window averages）
# ============================================================
class OverlapFeature(BaseFeature):
    """
    对每个通道，用一组相互重叠的短时间窗（默认 100 ms）对信号取平均，
    再把所有窗口、所有通道的均值拼接成一个特征向量。

    作用（对应论文 Figure 5）：
        - 100 ms 窗内平均相当于对时间维度做平滑/降采样，抑制高频抖动、
          突出 ERN/Pe 这类慢波；
        - 相邻窗口有 50 ms 重叠，保证波形变化不会因窗口边界而丢失。

    论文 Table 3 中该特征配合 sLDA 取得了最高平均准确率。
    """

    def __init__(self, sfreq, win_ms: float = 100.0, step_ms: float = 50.0):
        self.sfreq = sfreq
        self.win_len = int(round(win_ms / 1000.0 * sfreq))   # 窗口样本数
        self.step_len = int(round(step_ms / 1000.0 * sfreq))  # 滑动步长样本数

    def _fit(self, X, y):
        # 预先算好每个窗口的样本索引（对所有试次相同）
        n_times = X.shape[2]
        if n_times <= self.win_len:
            # 时间窗选择后区间可能短于 100ms：退化为覆盖整个区间的单个窗，
            # 保证在任意短区间上都能产出特征（对应对该区间整体取平均）
            self.windows = [(0, n_times)]
            return
        starts = range(0, n_times - self.win_len + 1, self.step_len)
        self.windows = [(s, s + self.win_len) for s in starts]

    def _apply(self, X):
        # 对每个窗口沿时间轴取平均：(n, ch, n_windows)
        means = np.stack([
            X[:, :, s:e].mean(axis=2) for s, e in self.windows
        ], axis=2)
        # 展平为 (n, ch * n_windows)
        return means.reshape(len(X), -1)


# ============================================================
# 工厂函数
# ============================================================
def build_feature(key: str, ch_names, sfreq, dataset_key: str = ""):
    """
    按特征 key 构造对应的特征提取器。

    参数：
        key         : 'FCzCz' / 'Timepoints' / 'Overlap' / 'xDAWN' / 'FCB'
        ch_names    : 数据集通道名
        sfreq       : 采样率
        dataset_key : 数据集 key，用于选择逐数据集空间成分数
    """
    if key == "FCzCz":
        return FCzCzFeature(ch_names)
    if key == "Timepoints":
        return TimepointsFeature()
    if key == "Overlap":
        return OverlapFeature(sfreq)
    if key == "xDAWN":
        n_comp = config.SPATIAL_COMPONENTS.get(dataset_key, 2)
        return XDawnFeature(ch_names, sfreq, n_components=n_comp)
    if key == "FCB":
        n_comp = config.SPATIAL_COMPONENTS.get(dataset_key, 2)
        return FCBFeature(n_components=n_comp)
    raise ValueError(f"未知特征 key：{key}")
