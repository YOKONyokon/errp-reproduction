# -*- coding: utf-8 -*-
"""
config.py
=========
ERRP 复现项目的全局配置：集中管理目录路径、信号处理参数、数据集信息与下载链接。

设计原则：
    所有“可改的超参数 / 路径 / URL”都放在这里，其余模块只读取本文件，
    避免在代码里散落硬编码（hard-coded）常量，便于维护与扩展新数据集。

术语：ERRP = Error-Related Potential（错误相关电位）
"""

from pathlib import Path

# ============================================================
# 1. 目录路径
# ============================================================
# 项目根目录（本文件所在目录）
ROOT_DIR = Path(__file__).resolve().parent

# 官方/第三方源码与数据目录
THIRD_PARTY_DIR = ROOT_DIR / "third_party"

# FCB 官方 Python 工具箱（来源：https://github.com/gpiresML/FCB-spatial-filter）
FCB_TOOLBOX_DIR = THIRD_PARTY_DIR / "FCB-spatial-filter" / "Python"

# HRI 数据集（来源：https://github.com/stefan-ehrlich/dataset-ErrP-HRI）
HRI_DIR = THIRD_PARTY_DIR / "dataset-ErrP-HRI"

# Coadaptation 数据集（来源：https://github.com/stefan-ehrlich/dataset-ErrP-coadaptation）
COADAPTATION_DIR = THIRD_PARTY_DIR / "dataset-ErrP-coadaptation"

# Virtual Cursor 数据集（原始 EEG 来自 GitHub Release 的 data_1.zip）
VIRTUAL_CURSOR_DIR = THIRD_PARTY_DIR / "virtual-cursor-release" / "data" / "raw_data"

# BNCI 013-2015 数据集（可靠直链来自 TU Graz lampx 镜像）
BNCI_DIR = THIRD_PARTY_DIR / "bnci-013"

# Gaze-Based Typesetting 数据集（figshare，真实浏览器通过 Cloudflare 下载）
GAZE_DIR = THIRD_PARTY_DIR / "gaze"

# NEMAR ERP CORE（nm000132，BIDS/EEGLAB，阶段 3 纳入；本阶段用 flankers/ERN 任务）
NEMAR_DIR = THIRD_PARTY_DIR / "nm000132"

# 结果输出目录（表格、图、JSON）
RESULTS_DIR = ROOT_DIR / "results"

# ============================================================
# 2. 信号处理与 epoch 参数（对齐论文口径）
# ============================================================
RANDOM_SEED = 42          # 随机种子，保证结果可复现

# 论文统一带通滤波范围（Yasemin 2023 默认 1–10 Hz）
BANDPASS_LOW_HZ = 1.0
BANDPASS_HIGH_HZ = 10.0

# 反馈锁定（feedback-locked）epoch 窗口，单位秒
# t=0 为反馈/错误结果呈现时刻；取反馈前 0.2 s 到反馈后 0.8 s
EPOCH_TMIN_S = -0.2
EPOCH_TMAX_S = 0.8

# 基线校正区间（用反馈前的安静段做基线）
BASELINE_TMIN_S = -0.2
BASELINE_TMAX_S = 0.0

# ============================================================
# 3. 类别标签编码（统一约定）
# ============================================================
# 错误（error / ErrP）记为 1，正确（correct）记为 0
LABEL_ERROR = 1
LABEL_CORRECT = 0

# ============================================================
# 4. 数据集信息与下载链接（每个数据集的来源都可追溯）
# ============================================================
# 数据集 A：LSC（FCB 官方工具箱自带的已预处理 ErrP 数据）
DATASET_LSC = {
    "key": "LSC",
    "name": "LSC speller ErrP (FCB 官方示例数据)",
    # 随 FCB 仓库一起提供的 .mat（12 通道、256 Hz、1 s、已 1–10 Hz 滤波）
    "mat_path": FCB_TOOLBOX_DIR / "data" / "ErrP_LSC_dataset.mat",
    "sfreq_hz": 256,
    "source_url": "https://github.com/gpiresML/FCB-spatial-filter",
    "origin_dataset_url":
        "https://ieee-dataport.org/open-access/"
        "error-related-potentials-primary-and-secondary-errp-and-p300-"
        "event-related-potentials-%E2%80%93",
    "license": "MIT (FCB 工具箱)；原始 LSC 数据见 IEEE DataPort",
    "already_preprocessed": True,   # 该数据已滤波、已切 epoch
}

# 数据集 B：HRI cursor（连续 raw，需自行滤波与切 epoch）
DATASET_HRI = {
    "key": "HRIcursor",
    "name": "HRI cursor · monitoring machine error (Ehrlich & Cheng 2019)",
    "data_dir": HRI_DIR / "cursor",
    "sfreq_hz": 256,
    "source_url": "https://github.com/stefan-ehrlich/dataset-ErrP-HRI",
    "origin_paper_url": "https://doi.org/10.1007/s12369-018-0501-8",
    "license": "见数据集文档（CC BY-NC-ND，BNCI/作者条款）",
    "already_preprocessed": False,
    # EEGLAB 事件码（含义见文档 Trigger/Marker information）
    "event_correct": "S  4",   # Feedback no error
    "event_human_error": "S  5",
    "event_machine_error": "S  6",
    "n_eeg_channels": 27,      # 30 通道中含 3 个 EOG，其余 27 个为 EEG
}

# ============================================================
# 数据集 C：HRI robot（与 cursor 同仓库、同事件口径，仅反馈载体为 NAO 机器人）
# ============================================================
DATASET_HRI_ROBOT = {
    "key": "HRIrobot",
    "name": "HRI robot · validating robot actions (Ehrlich & Cheng 2018)",
    "data_dir": HRI_DIR / "robot",
    "file_glob": "*_robot.set",
    "sfreq_hz": 256,
    "source_url": "https://github.com/stefan-ehrlich/dataset-ErrP-HRI",
    "origin_paper_url": "https://doi.org/10.1007/s12369-018-0501-8",
    "license": "见数据集文档（CC BY-NC-ND，作者条款）",
    "already_preprocessed": False,
    # 事件含义与 cursor 完全一致（见 documentation_dataset-ErrP-HRI.pdf）
    "event_correct": "S  4",   # Feedback no error
    "event_machine_error": "S  6",
    "n_eeg_channels": 27,
}

# ============================================================
# 数据集 D：Coadaptation（人机协同自适应，EEGLAB 数字事件码）
# ============================================================
DATASET_COADAPTATION = {
    "key": "Coadaptation",
    "name": "Human-agent co-adaptation (Ehrlich & Cheng 2018)",
    "data_dir": COADAPTATION_DIR / "data_coadaptation",
    "sfreq_hz": 256,
    "source_url": "https://github.com/stefan-ehrlich/dataset-ErrP-coadaptation",
    "origin_paper_url": "https://doi.org/10.1088/1741-2552/aae069",
    "license": "见数据集文档（作者条款）",
    "already_preprocessed": False,
    # 官方 Dataset5_savemat 口径：13 个被试、calib + corl1/2/4
    "subjects": [3, 4, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18],
    "sessions": ["calib", "corl1", "corl2", "corl4"],
    # EEGLAB 数字 marker（含义见 Dataset documentation Co-adaptation.pdf）
    "event_correct_code": 33031,   # Feedback no error
    "event_error_code": 33033,     # Feedback machine/human error
    "n_eeg_channels": 27,
}

# ============================================================
# 数据集 E：BNCI 013-2015 monitoring ErrP（TU Graz 可靠直链）
# ============================================================
DATASET_BNCI = {
    "key": "BNCI013",
    "name": "BNCI Horizon 2020 · 013-2015 monitoring ErrP",
    "data_dir": BNCI_DIR,
    "sfreq_hz": 512,
    "source_url": "https://lampx.tugraz.at/~bci/database/013-2015/",
    "origin_paper_url": "https://doi.org/10.1109/TNSRE.2010.2053387",
    "license": "CC BY-NC-ND 4.0",
    "already_preprocessed": False,
    "subjects": [1, 2, 3, 4, 5, 6],
    "runs": ["s1", "s2"],
    # 事件码（见 MOABB _convert_run_epfl）：5/10=正确，6/9=错误
    "event_correct_codes": [5, 10],
    "event_error_codes": [6, 9],
    "n_eeg_channels": 64,
}

# ============================================================
# 数据集 F：Virtual Cursor（原始 EEG 来自 GitHub Release，16 通道）
# ============================================================
DATASET_VIRTUAL_CURSOR = {
    "key": "VirtualCursor",
    "name": "Virtual Cursor · self-calibration BCI (Iturrate et al. 2015)",
    "data_dir": VIRTUAL_CURSOR_DIR,
    "sfreq_hz": 256,
    "source_url":
        "https://github.com/flowersteam/self_calibration_BCI_plosOne_2015",
    "origin_paper_url": "https://doi.org/10.1371/journal.pone.0131491",
    "license": "可无限制使用（见 raw_data/readme.txt）",
    "already_preprocessed": False,
    "subjects": [1, 2, 3, 4, 5, 6, 7, 8],
    # 事件名（见 raw_data/readme.txt）
    "event_correct_name": "correct_movement",
    "event_error_name": "error_movement",
    "n_eeg_channels": 16,
}

# ============================================================
# 数据集 G：Gaze-Based Typesetting（figshare，10 被试，61 通道，已切 epoch）
# ============================================================
DATASET_GAZE = {
    "key": "Gaze",
    "name": "Error Related Potentials from Gaze-Based Typesetting (Kalaganis 2018)",
    "data_dir": GAZE_DIR,
    "sfreq_hz": 256,
    "source_url":
        "https://figshare.com/articles/dataset/"
        "Error_Related_Potentials_from_Gaze-Based_Typesetting/5938714/1",
    "doi_url": "https://doi.org/10.6084/m9.figshare.5938714.v1",
    "origin_paper_url": "https://zenodo.org/records/1296002",
    "license": "CC BY 4.0",
    # 数据已做 zero phase 带通 1–20Hz 且已切好 epoch，无需再滤波
    "already_preprocessed": True,
    "prefilter_band": (1.0, 20.0),
    # 被试编号不连续（数据本身如此）：S01–S07、S11、S13、S15
    "subjects": ["S01", "S02", "S03", "S04", "S05", "S06",
                 "S07", "S11", "S13", "S15"],
    "tmin_s": -0.2,          # 180 点 / 256Hz ≈ 0.703s，窗口 [-0.2, 0.5]
    "tmax_s": 0.5,
    "n_eeg_channels": 61,
}

# 汇总：本次复现实际运行的数据集清单（7 个可获取数据集）
# ============================================================
# 数据集 H：NEMAR ERP CORE · flankers ERN（反应锁定，阶段 3 纳入）
# ============================================================
DATASET_ERPCORE_ERN = {
    "key": "ERPCORE_ERN",
    "name": "ERP CORE Eriksen flankers · ERN (Kappenman 2021, NEMAR nm000132)",
    "data_dir": NEMAR_DIR,
    "source_url": "https://nemar.org/dataset/nm000132",
    "origin_paper_url": "https://doi.org/10.1016/j.neuroimage.2020.117465",
    "license": "CC-BY-4.0",
    "already_preprocessed": False,
    # 原始 1024 Hz、30 EEG + 3 EOG，加载后降采样到 256
    "raw_sfreq_hz": 1024,
    "target_sfreq_hz": 256,
    "n_eeg_channels": 30,
    # 反应事件三位码：末位 1=正确、2=错误
    "response_correct_codes": [111, 121, 212, 222],
    "response_error_codes": [112, 122, 211, 221],
    # 反应锁定窗口（response-locked），区别于反馈锁定数据集
    "response_tmin_s": -0.2,
    "response_tmax_s": 0.6,
}

ALL_DATASETS = [
    DATASET_LSC,
    DATASET_HRI,
    DATASET_HRI_ROBOT,
    DATASET_COADAPTATION,
    DATASET_BNCI,
    DATASET_VIRTUAL_CURSOR,
    DATASET_GAZE,
    DATASET_ERPCORE_ERN,
]

# ============================================================
# 5. 方法矩阵：特征 × 分类器
# ============================================================
# 五种特征提取方法（对应论文 3.2.5 节 Feature extraction）：
#   FCzCz     : 中线电极 FCz/Cz（无 FCz 时用 Fz/Cz）
#   Timepoints: 全电极时间点，不做空间变换
#   Overlap   : 重叠窗口平均（100 ms 窗、50 ms 步长）
#   xDAWN     : xDAWN 空间滤波（MNE 官方）
#   FCB       : FCB 空间滤波（官方 Python 工具箱）
FEATURE_KEYS = ["FCzCz", "Timepoints", "Overlap", "xDAWN", "FCB"]

# 四种分类器（对应论文 MATLAB 中 MVPA-Light 的分类器）
CLASSIFIER_KEYS = ["GNB", "sLDA", "SVM", "RBF-SVM"]

# 置换检验（permutation test）次数
N_PERMUTATIONS = 1000

# 交叉验证折数
CV_FOLDS = 5

# 逐数据集空间滤波成分数（xDAWN/FCB 通用），取自官方 ErrP_APP document.xml
# 中每个数据集的 scomp（论文 3.2.5：成分数在 1–4 之间，逐数据集取最佳）
SPATIAL_COMPONENTS = {
    "LSC": 2,
    "HRIcursor": 4,
    "HRIrobot": 4,
    "Coadaptation": 4,
    "BNCI013": 4,
    "Gaze": 4,
    "VirtualCursor": 2,
    "ERPCORE_ERN": 4,
}


def ensure_dirs():
    """运行前确保结果目录存在。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
