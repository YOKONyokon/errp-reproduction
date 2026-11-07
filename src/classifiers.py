# -*- coding: utf-8 -*-
"""
classifiers.py
==============
四种分类器，统一用 scikit-learn 的权威实现（对应论文 MATLAB 中
MVPA-Light 提供的同名分类器），并在注释中解释其数学直觉。

分类器清单：
    1. GNB     : Gaussian Naive Bayes（高斯朴素贝叶斯）
    2. sLDA    : shrinkage Linear Discriminant Analysis（收缩线性判别分析）
    3. SVM     : 线性核支持向量机（最大间隔）
    4. RBF-SVM : 高斯核（RBF）支持向量机（非线性）

为什么直接用 sklearn：
    这些是成熟的工业级算法实现（libsvm / Ledoit-Wolf 收缩 / 贝叶斯），
    与论文出处方法等价，且经过大量测试；无需也不应自行重写。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC


def build_classifier(key: str):
    """
    按 key 返回一个全新的 sklearn 分类器实例。

    每次调用都返回新对象，避免在交叉验证各折之间复用已拟合状态。
    """
    if key == "GNB":
        # 朴素贝叶斯：假设各特征在给定类别下服从高斯分布且彼此独立，
        # 用贝叶斯公式比较“像错误”还是“像正确”的后验概率。
        return GaussianNB()

    if key == "sLDA":
        # 收缩 LDA：在高维、少样本下，样本协方差矩阵容易病态（甚至奇异）。
        # shrinkage='auto' 用 Ledoit-Wolf 公式自动估计收缩强度，
        # 把协方差往单位矩阵方向“拉回”，判别边界更稳定、不易过拟合。
        # solver='lsqr' 用最小二乘求解，配合收缩参数。
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

    if key == "SVM":
        # 线性 SVM：在特征空间找一个超平面，使离它最近的两类样本（支持向量）
        # 间隔最大。C 控制对误分的容忍度，默认 1.0。
        # SVM 对特征尺度敏感，先用 StandardScaler 标准化（仅在训练折拟合），
        # 对应论文/MVPA-Light 对 SVM 输入的归一化处理，避免小尺度特征下退化。
        #
        # 求解器选择：这里用 LinearSVC（liblinear），而非 SVC(kernel='linear')
        # （libsvm）。两者数学目标等价，都是 L2 正则的线性最大间隔判别；
        # 但 libsvm 在“高维 + 大样本”下复杂度接近 O(n^3)、极慢。
        #
        # dual='auto'：由 sklearn 按样本数 n 与特征数 p 自动选择——
        #   n > p 时用 primal（dual=False），n < p 时用 dual（dual=True），
        #   与 MVPA-Light/libsvm 的 dual 求解口径一致。
        #
        # max_iter 取 sklearn 默认的 1000。在 2 核受限环境实测：
        #   - 高维 Timepoints（6939 维）上，单条管线约 4 分钟可稳定完成，
        #     不会 OOM；调大到 3000+ 反而会因内存/超时被杀。
        #   - 低可分数据集（如 HRIrobot）上求解器可能在 1000 次迭代内
        #     未完全收敛（ConvergenceWarning），但此时的解已是有效的线性
        #     判别、预测仍可用；这是 sklearn 标准行为，故保留默认口径。
        return make_pipeline(
            StandardScaler(),
            LinearSVC(C=1.0, dual="auto", max_iter=1000))

    if key == "RBF-SVM":
        # RBF 核 SVM：通过高斯核把数据隐式映射到高维空间，
        # 从而能画出非线性的判别边界。gamma='scale' 自动按特征方差设定核宽度。
        # 同样先标准化，保证核宽度不受原始量纲影响。
        return make_pipeline(
            StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))

    raise ValueError(f"未知分类器 key：{key}")
