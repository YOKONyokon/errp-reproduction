# -*- coding: utf-8 -*-
"""
data_loader.py
==============
数据加载器：把两个不同来源的真实 ErrP 数据统一成相同的内存结构，
供后续特征提取与分类使用，体现“同一套管线、不同数据集”的复现思想。

统一输出容器 EpochData：
    X        : (n_trials, n_channels, n_times)  试次在前的三维数组
    y        : (n_trials,)                      0=正确, 1=错误
    ch_names : 通道名列表
    sfreq    : 采样率（Hz）
    subjects : 每个试次所属被试的标识数组

数据集：
    - load_lsc()      : 读取 FCB 官方自带的 LSC .mat（已预处理）
    - load_hri()      : 用 MNE 读取 HRI 连续 raw 并自行滤波/切 epoch
    - load_dataset()  : 按 key 统一调度
"""

import sys
from dataclasses import dataclass
from pathlib import Path

# 把项目根目录加入模块搜索路径，保证无论从哪里运行都能找到 config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mne
import numpy as np
from scipy.io import loadmat

import config

# 让本目录模块可被导入
from preprocessing import butter_bandpass, baseline_correct


# ============================================================
# 统一数据容器
# ============================================================
@dataclass
class EpochData:
    """存放一个数据集的全部 epoch 及其元信息。"""
    X: np.ndarray            # (n_trials, n_channels, n_times)
    y: np.ndarray            # (n_trials,)
    ch_names: list           # 通道名
    sfreq: float             # 采样率
    subjects: np.ndarray     # (n_trials,) 被试标识
    tmin: float = -0.2       # 首个时间点相对事件的时间（秒）

    def summary(self) -> str:
        """返回人类可读的数据集摘要字符串。"""
        n_err = int(np.sum(self.y == config.LABEL_ERROR))
        n_cor = int(np.sum(self.y == config.LABEL_CORRECT))
        return (f"试次={len(self.y)} (错误 {n_err} / 正确 {n_cor}), "
                f"通道={len(self.ch_names)}, 采样率={self.sfreq} Hz, "
                f"时间点={self.X.shape[2]}, 被试数={len(set(self.subjects))}")


# LSC 数据的 12 通道固定顺序（来源：FCB 官方 main.py 注释）
LSC_CH_NAMES = ["Fz", "Cz", "C3", "C4", "CPz", "Pz",
                "P3", "P4", "PO7", "PO8", "POz", "Oz"]


# ============================================================
# 数据集 A：LSC（FCB 官方 .mat，已滤波、已切 epoch）
# ============================================================
def load_lsc() -> EpochData:
    """
    加载 LSC ErrP 数据。

    .mat 结构（scipy 读取嵌套 struct）：
        ErrP    : (12, 256, 28)   错误试次（通道×时间×试次）
        CorrERP : (12, 256, 162)  正确试次
    """
    info = config.DATASET_LSC
    mat = loadmat(info["mat_path"])
    struct = mat["ErrP_LSC_dataset"][0, 0]
    err = struct["ErrP"]       # (12, 256, 28)
    cor = struct["CorrERP"]    # (12, 256, 162)

    # 转成“试次在前”：(n_trials, n_channels, n_times)
    err = np.transpose(err, axes=(2, 0, 1)).astype(np.float64)
    cor = np.transpose(cor, axes=(2, 0, 1)).astype(np.float64)

    # 拼接两类并构造标签：错误=1，正确=0
    X = np.concatenate([err, cor], axis=0)
    y = np.concatenate([
        np.full(len(err), config.LABEL_ERROR),
        np.full(len(cor), config.LABEL_CORRECT),
    ])

    # LSC 为 FCB 官方示例，被试统一标记为 "LSC-pool"
    subjects = np.array(["LSC-pool"] * len(y))
    return EpochData(X=X, y=y, ch_names=list(LSC_CH_NAMES),
                     sfreq=info["sfreq_hz"], subjects=subjects)


# ============================================================
# 数据集 B：HRI cursor（连续 raw，MNE 官方接口处理）
# ============================================================
def _load_one_hri_subject(set_path, subject_id: str) -> EpochData:
    """
    读取单个 HRI 被试的连续数据，完成：剔除 EOG → 1–10Hz 带通
    → 按 S4/S6 反馈事件切 epoch → 基线校正。

    事件口径（见数据集文档）：
        S  4 = Feedback no error（正确）
        S  6 = Feedback machine error（机器错误，monitoring 关注）
    """
    info = config.DATASET_HRI

    # 1) 读取连续 EEGLAB 数据并预加载到内存
    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose="ERROR")

    # 2) 按通道名剔除 EOG（EEGLAB 中 EOG 也被标成 eeg 类型，需手动剔除）
    eog_names = [ch for ch in raw.ch_names if "EOG" in ch.upper()]
    if eog_names:
        raw.drop_channels(eog_names)

    # 3) MNE 官方带通滤波 1–10 Hz（FIR、零相位，自动处理边界）
    raw.filter(config.BANDPASS_LOW_HZ, config.BANDPASS_HIGH_HZ,
               verbose="ERROR")

    # 4) 从注释提取事件，并只保留正确/机器错误两类
    events, event_map = mne.events_from_annotations(raw, verbose="ERROR")
    event_id = {
        "correct": event_map[info["event_correct"]],
        "error": event_map[info["event_machine_error"]],
    }

    # 5) 切反馈锁定 epoch 并做基线校正
    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=config.EPOCH_TMIN_S, tmax=config.EPOCH_TMAX_S,
        baseline=(config.BASELINE_TMIN_S, config.BASELINE_TMAX_S),
        preload=True, verbose="ERROR",
    )

    X = epochs.get_data(copy=True).astype(np.float64)   # (n, ch, t)
    # MNE 事件码转回统一标签：error -> 1, correct -> 0
    code_correct, code_error = event_id["correct"], event_id["error"]
    y = np.where(epochs.events[:, 2] == code_error,
                 config.LABEL_ERROR, config.LABEL_CORRECT)
    subjects = np.array([subject_id] * len(y))
    return EpochData(X=X, y=y, ch_names=list(epochs.ch_names),
                     sfreq=info["sfreq_hz"], subjects=subjects)


def load_hri(max_subjects: int = None) -> EpochData:
    """
    加载并合并 HRI cursor 的多个被试。

    参数：
        max_subjects : 最多加载多少个被试（None=全部）。
                       用于快速冒烟测试时只取前 1–2 个被试。
    """
    data_dir = config.DATASET_HRI["data_dir"]
    set_files = sorted(data_dir.glob("*_cursor.set"))
    if max_subjects is not None:
        set_files = set_files[:max_subjects]
    if not set_files:
        raise FileNotFoundError(f"未找到 HRI set 文件：{data_dir}")

    per_subject = []
    for set_path in set_files:
        # 文件名形如 s02_cursor.set，提取 s02 作为被试 id
        subject_id = set_path.name.split("_")[0]
        per_subject.append(_load_one_hri_subject(set_path, subject_id))

    # HRI 各被试通道一致，可直接沿试次轴拼接
    X = np.concatenate([d.X for d in per_subject], axis=0)
    y = np.concatenate([d.y for d in per_subject], axis=0)
    subjects = np.concatenate([d.subjects for d in per_subject], axis=0)
    return EpochData(X=X, y=y, ch_names=per_subject[0].ch_names,
                     sfreq=per_subject[0].sfreq, subjects=subjects)


# ============================================================
# 通用 EEGLAB 加载器：同时服务 HRIcursor / HRIrobot / Coadaptation
# ============================================================
def _epochs_from_eeglab(set_path, subject_id: str,
                        correct_desc: str, error_desc: str,
                        sfreq_hz: float) -> EpochData:
    """
    读取单个 EEGLAB .set/.fdt 连续记录，完成统一预处理：
        剔除 EOG → 1–10Hz 带通 → 按反馈事件切 epoch → 基线校正。

    参数：
        correct_desc / error_desc : 正确/错误事件在 annotation 中的描述。
            HRI 用字符串 'S  4'/'S  6'；Coadaptation 用数字串 '33031'/'33033'。
    """
    # 1) 读取连续 EEGLAB 数据并预加载
    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose="ERROR")

    # 2) 剔除 EOG 通道（EEGLAB 里 EOG 也被标成 eeg，需按名称剔除）
    eog_names = [ch for ch in raw.ch_names if "EOG" in ch.upper()]
    if eog_names:
        raw.drop_channels(eog_names)

    # 3) 统一 1–10Hz 带通（FIR、零相位）
    raw.filter(config.BANDPASS_LOW_HZ, config.BANDPASS_HIGH_HZ,
               verbose="ERROR")

    # 4) 从注释提取事件，只保留正确/错误两类
    events, event_map = mne.events_from_annotations(raw, verbose="ERROR")
    # 若某类事件在该文件缺失则给出明确报错，避免静默漏类
    for desc in (correct_desc, error_desc):
        if desc not in event_map:
            raise KeyError(f"事件 {desc} 未出现在 {set_path.name}，可用：{list(event_map)}")
    event_id = {
        "correct": event_map[correct_desc],
        "error": event_map[error_desc],
    }

    # 5) 切反馈锁定 epoch 并做基线校正（全数据集统一窗口，保证可比）
    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=config.EPOCH_TMIN_S, tmax=config.EPOCH_TMAX_S,
        baseline=(config.BASELINE_TMIN_S, config.BASELINE_TMAX_S),
        preload=True, verbose="ERROR",
    )

    X = epochs.get_data(copy=True).astype(np.float64)
    y = np.where(epochs.events[:, 2] == event_id["error"],
                 config.LABEL_ERROR, config.LABEL_CORRECT)
    subjects = np.array([subject_id] * len(y))
    return EpochData(X=X, y=y, ch_names=list(epochs.ch_names),
                     sfreq=sfreq_hz, subjects=subjects)


def _merge_epochs(per_file: list) -> EpochData:
    """把多个文件/被试的 EpochData 沿试次轴拼接（要求通道与采样率一致）。"""
    X = np.concatenate([d.X for d in per_file], axis=0)
    y = np.concatenate([d.y for d in per_file], axis=0)
    subjects = np.concatenate([d.subjects for d in per_file], axis=0)
    return EpochData(X=X, y=y, ch_names=per_file[0].ch_names,
                     sfreq=per_file[0].sfreq, subjects=subjects)


# ============================================================
# 数据集 C：HRI robot（与 cursor 同构，仅文件 glob 与反馈载体不同）
# ============================================================
def load_hri_robot(max_subjects: int = None) -> EpochData:
    """加载 HRI robot 全部被试，事件口径 S4=正确、S6=机器错误。"""
    info = config.DATASET_HRI_ROBOT
    set_files = sorted(info["data_dir"].glob(info["file_glob"]))
    if max_subjects is not None:
        set_files = set_files[:max_subjects]
    if not set_files:
        raise FileNotFoundError(f"未找到 HRI robot 文件：{info['data_dir']}")

    per_file = []
    for set_path in set_files:
        subject_id = set_path.name.split("_")[0]
        per_file.append(_epochs_from_eeglab(
            set_path, subject_id,
            correct_desc=info["event_correct"],
            error_desc=info["event_machine_error"],
            sfreq_hz=info["sfreq_hz"]))
    return _merge_epochs(per_file)


# ============================================================
# 数据集 D：Coadaptation（数字事件码，13 被试 × calib/corl1/2/4）
# ============================================================
def load_coadaptation(max_subjects: int = None) -> EpochData:
    """
    按官方 Dataset5_savemat 口径加载 Coadaptation：
        13 个被试，每个被试拼接 calib + corl1/corl2/corl4，
        事件 33031=正确、33033=错误。
    """
    info = config.DATASET_COADAPTATION
    subject_ids = info["subjects"]
    if max_subjects is not None:
        subject_ids = subject_ids[:max_subjects]

    per_file = []
    for sub in subject_ids:
        sdir = info["data_dir"] / f"s{sub:02d}"
        for session in info["sessions"]:
            set_path = sdir / f"s{sub:02d}_{session}.set"
            if not set_path.exists():
                # 个别被试可能缺某个 session，跳过并提示
                print(f"  - 缺少文件，跳过：{set_path.name}")
                continue
            per_file.append(_epochs_from_eeglab(
                set_path, f"s{sub:02d}",
                correct_desc=str(info["event_correct_code"]),
                error_desc=str(info["event_error_code"]),
                sfreq_hz=info["sfreq_hz"]))
    if not per_file:
        raise FileNotFoundError("Coadaptation 未加载到任何文件")
    return _merge_epochs(per_file)


# ============================================================
# 通用 numpy → MNE Raw → epoch（服务 BNCI 与 Virtual Cursor 的 .mat）
# ============================================================
def _epochs_from_numpy(eeg_uV, ch_names, sfreq,
                       pos_code_pairs, subject_id) -> EpochData:
    """
    把裸 numpy 连续 EEG 与事件位置/类型包装成 MNE Raw，再统一处理。

    参数：
        eeg_uV         : (n_samples, n_channels)，单位 µV
        pos_code_pairs : [(pos, code)]，code 1=正确、2=错误；pos 为 1-based
    """
    from mne import create_info
    from mne.io import RawArray

    n_samples = eeg_uV.shape[0]
    # 1) 构造 STI 触发通道（在事件位置写 1/2），与 MOABB 做法一致
    trigger = np.zeros((n_samples, 1), dtype=np.float64)
    for pos, code in pos_code_pairs:
        trigger[pos - 1, 0] = code      # MATLAB 1-based → numpy 0-based

    # 2) µV → V，拼接 STI 通道，构造 RawArray
    eeg_v = eeg_uV.astype(np.float64) * 1e-6
    data = np.c_[eeg_v, trigger].T
    all_names = list(ch_names) + ["STI"]
    all_types = ["eeg"] * len(ch_names) + ["stim"]
    info = create_info(ch_names=all_names, ch_types=all_types, sfreq=sfreq)
    raw = RawArray(data, info, verbose="ERROR")

    # 3) 统一 1–10Hz 带通（MNE 自动只作用于 EEG，不影响 STI）
    raw.filter(config.BANDPASS_LOW_HZ, config.BANDPASS_HIGH_HZ,
               verbose="ERROR")

    # 4) 从 STI 通道读事件，切统一 epoch 并基线校正
    events = mne.find_events(raw, stim_channel="STI", verbose="ERROR")
    event_id = {"correct": 1, "error": 2}
    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=config.EPOCH_TMIN_S, tmax=config.EPOCH_TMAX_S,
        baseline=(config.BASELINE_TMIN_S, config.BASELINE_TMAX_S),
        preload=True, verbose="ERROR",
    )

    # 关键防泄漏：STI 通道在 t=0 处直接编码了类标签（1/2），
    # 若把它当特征，任何分类器都能“读标签”得到虚假的 100%。
    # 切完 epoch 后立即丢弃 STI，只保留纯 EEG 通道。
    epochs.drop_channels("STI")

    X = epochs.get_data(copy=True).astype(np.float64)
    y = np.where(epochs.events[:, 2] == 2,
                 config.LABEL_ERROR, config.LABEL_CORRECT)
    subjects = np.array([subject_id] * len(y))
    return EpochData(X=X, y=y, ch_names=list(ch_names),
                     sfreq=sfreq, subjects=subjects)


# ============================================================
# 数据集 E：BNCI 013-2015（run/eeg/header/EVENT，6 被试 × 2 run）
# ============================================================
def load_bnci(max_subjects: int = None) -> EpochData:
    """加载 BNCI 013-2015，事件 5/10=正确、6/9=错误。"""
    info = config.DATASET_BNCI
    subject_ids = info["subjects"]
    if max_subjects is not None:
        subject_ids = subject_ids[:max_subjects]

    correct_codes = set(info["event_correct_codes"])
    error_codes = set(info["event_error_codes"])
    per_file = []

    for sub in subject_ids:
        ch_names = None
        for run in info["runs"]:
            mat_path = info["data_dir"] / f"Subject{sub:02d}_{run}.mat"
            if not mat_path.exists():
                print(f"  - 缺少文件，跳过：{mat_path.name}")
                continue
            mat = loadmat(str(mat_path), struct_as_record=False,
                          squeeze_me=True)
            for rec in mat["run"]:
                header = rec.header
                # Label 最后一个是 'Status'，只取与 eeg 列数相等的前 64 个
                ch_names = list(header.Label[:rec.eeg.shape[1]])
                pairs = []
                for pos, typ in zip(header.EVENT.POS, header.EVENT.TYP):
                    if typ in correct_codes:
                        pairs.append((int(pos), 1))
                    elif typ in error_codes:
                        pairs.append((int(pos), 2))
                per_file.append(_epochs_from_numpy(
                    rec.eeg, ch_names, float(header.SampleRate),
                    pairs, f"s{sub:02d}"))
    if not per_file:
        raise FileNotFoundError("BNCI 未加载到任何文件")
    return _merge_epochs(per_file)


# ============================================================
# 数据集 F：Virtual Cursor（EEG.signal/events，8 被试，16 通道）
# ============================================================
def load_virtual_cursor(max_subjects: int = None) -> EpochData:
    """加载 Virtual Cursor，事件 correct_movement/error_movement。"""
    info = config.DATASET_VIRTUAL_CURSOR
    subject_ids = info["subjects"]
    if max_subjects is not None:
        subject_ids = subject_ids[:max_subjects]

    # 16 通道固定顺序（来源：electrode_locations.txt）
    ch16 = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C3", "C1",
            "Cz", "C2", "C4", "CP3", "CP1", "CPz", "CP2", "CP4"]
    per_file = []

    for sub in subject_ids:
        mat_path = info["data_dir"] / f"EEG_s{sub}.mat"
        if not mat_path.exists():
            print(f"  - 缺少文件，跳过：{mat_path.name}")
            continue
        mat = loadmat(str(mat_path), struct_as_record=False,
                      squeeze_me=True)
        eeg = mat["EEG"]
        names = eeg.events.name
        positions = eeg.events.position
        pairs = []
        for name, pos in zip(names, positions):
            if name == info["event_correct_name"]:
                pairs.append((int(pos), 1))
            elif name == info["event_error_name"]:
                pairs.append((int(pos), 2))
        per_file.append(_epochs_from_numpy(
            eeg.signal, ch16, float(eeg.sample_rate), pairs, f"s{sub}"))
    if not per_file:
        raise FileNotFoundError("Virtual Cursor 未加载到任何文件")
    return _merge_epochs(per_file)


# ============================================================
# 数据集 G：Gaze-Based Typesetting（已切 epoch，10 被试，61 通道）
# ============================================================
def _clean_mat_name(raw) -> str:
    """
    把 namesEBN 里形如 "['AF7']" 的字符串清理成标准通道名 "AF7"。
    兼容已经是 'AF7' 或带方括号/引号的多种写法。
    """
    import ast
    import re
    s = str(np.squeeze(raw))
    # 优先尝试字面量解析（"['AF7']" -> ['AF7']）
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return str(v[0])
        return str(v)
    except Exception:
        pass
    m = re.search(r"([A-Za-z0-9]+)", s)
    return m.group(1) if m else s


def load_gaze(max_subjects: int = None) -> EpochData:
    """
    加载 Gaze-Based Typesetting 数据。

    数据已在发布前完成：
        zero phase 带通 1–20Hz、按反馈事件切好 epoch（256Hz、[-0.2,0.5]）。
    因此这里不再滤波（避免在 epoch 边界二次引入伪影），只做统一基线校正。

    .mat 结构（每个被试一个 struct，含 3 字段）：
        eeg_trials : (n_trials, 61, 180)   已切好的单试次脑电
        eye_trials : (n_trials, 2, 22)     眼动坐标（本复现不使用）
        labels     : (1, n_trials)         -1=错误按键，1=正确
    """
    info = config.DATASET_GAZE
    data_dir = info["data_dir"]
    subject_ids = info["subjects"]
    if max_subjects is not None:
        subject_ids = subject_ids[:max_subjects]

    # 1) 读取通道名（namesEBN 指示 eeg_trials 的通道顺序）
    nm = loadmat(str(data_dir / "namesEBN.mat"))
    ch_names = [_clean_mat_name(x) for x in nm["namesEBN"]]

    # 基线区间对应的时间点索引：tmin(-0.2) 到 t=0，256Hz 下约 0..51
    n_base = int(round(abs(info["tmin_s"]) * info["sfreq_hz"]))
    baseline_idx = np.arange(0, n_base + 1)

    per_subject = []
    for sub in subject_ids:
        mat = loadmat(str(data_dir / f"{sub}.mat"),
                      struct_as_record=False, squeeze_me=False)
        rec = mat[sub][0, 0]
        X = np.asarray(rec.eeg_trials, dtype=np.float64)   # (n,61,180)
        raw_labels = np.squeeze(np.asarray(rec.labels))    # (n,) -1/1

        # 2) 统一基线校正（减去反馈前安静段均值）
        X = baseline_correct(X, baseline_idx)

        # 3) 标签转换：-1(错误) -> 1，+1(正确) -> 0
        y = np.where(raw_labels == -1,
                     config.LABEL_ERROR, config.LABEL_CORRECT)
        subjects = np.array([sub] * len(y))
        per_subject.append(EpochData(X=X, y=y, ch_names=ch_names,
                                     sfreq=info["sfreq_hz"],
                                     subjects=subjects))
    return _merge_epochs(per_subject)


# ============================================================
# 数据集 H：NEMAR ERP CORE flankers ERN（反应锁定）
# ============================================================
def load_erp_core_ern(max_subjects: int = None) -> EpochData:
    """
    加载 NEMAR ERP CORE flankers 的反应锁定 ERN 数据。

    与反馈锁定数据集不同：这里以被试的按键反应为锚点（response-locked），
    按反应事件三位码的末位判断对错（1=正确、2=错误）。原始 1024 Hz、
    30 EEG + 3 EOG，加载时去 EOG、1–10 Hz 带通、切反应锁定 epoch、
    基线校正，再降采样到 256 Hz，与其余数据集口径统一。
    """
    info = config.DATASET_ERPCORE_ERN
    base = info["data_dir"]
    subject_dirs = sorted(
        p.name for p in base.glob("sub-*")
        if (p / "eeg" / f"{p.name}_task-flankers_eeg.set").exists())
    if max_subjects is not None:
        subject_dirs = subject_dirs[:max_subjects]
    if not subject_dirs:
        raise FileNotFoundError(f"未找到 ERP CORE flankers 数据：{base}")

    correct = set(info["response_correct_codes"])
    error = set(info["response_error_codes"])

    per_subject = []
    for sdir in subject_dirs:
        set_path = base / sdir / "eeg" / f"{sdir}_task-flankers_eeg.set"
        raw = mne.io.read_raw_eeglab(
            str(set_path), preload=True, verbose="ERROR")

        # 1) 剔除 EOG
        eog = [ch for ch in raw.ch_names if "EOG" in ch.upper()]
        if eog:
            raw.drop_channels(eog)

        # 2) 1–10 Hz 带通（与其他数据集统一）
        raw.filter(config.BANDPASS_LOW_HZ, config.BANDPASS_HIGH_HZ,
                   verbose="ERROR")

        # 3) 只保留正确/错误反应事件，重映射为统一事件 id
        events, event_map = mne.events_from_annotations(raw, verbose="ERROR")
        sel_events = []
        for desc, evid in event_map.items():
            code = int(desc)
            if code in correct or code in error:
                new_id = 100 if code in correct else 200
                for i in np.where(events[:, 2] == evid)[0]:
                    sel_events.append([int(events[i, 0]), 0, new_id])
        if not sel_events:
            raise KeyError(f"{sdir} 未找到反应事件")
        sel_events = np.asarray(sel_events, dtype=int)
        sel_events = sel_events[np.argsort(sel_events[:, 0])]

        # 4) 反应锁定 epoch + 基线校正
        event_id = {"correct": 100, "error": 200}
        epochs = mne.Epochs(
            raw, sel_events, event_id=event_id,
            tmin=info["response_tmin_s"], tmax=info["response_tmax_s"],
            baseline=(config.BASELINE_TMIN_S, config.BASELINE_TMAX_S),
            preload=True, verbose="ERROR")

        # 5) 降采样到 256 Hz
        epochs.resample(info["target_sfreq_hz"], verbose="ERROR")

        X = epochs.get_data(copy=True).astype(np.float64)
        y = np.where(epochs.events[:, 2] == event_id["error"],
                     config.LABEL_ERROR, config.LABEL_CORRECT)
        subjects = np.array([sdir] * len(y))
        per_subject.append(EpochData(
            X=X, y=y, ch_names=list(epochs.ch_names),
            sfreq=info["target_sfreq_hz"], subjects=subjects))
    return _merge_epochs(per_subject)


# ============================================================
# 统一调度入口
# ============================================================
def load_dataset(key: str, max_subjects: int = None) -> EpochData:
    """
    按数据集 key 加载对应数据。

    支持的 key：LSC、HRIcursor、HRIrobot、Coadaptation、BNCI013、
                VirtualCursor、Gaze、ERPCORE_ERN。
    max_subjects 对多被试数据集生效，用于快速冒烟测试。
    """
    if key == config.DATASET_LSC["key"]:
        return load_lsc()
    if key == config.DATASET_HRI["key"]:
        return load_hri(max_subjects=max_subjects)
    if key == config.DATASET_HRI_ROBOT["key"]:
        return load_hri_robot(max_subjects=max_subjects)
    if key == config.DATASET_COADAPTATION["key"]:
        return load_coadaptation(max_subjects=max_subjects)
    if key == config.DATASET_BNCI["key"]:
        return load_bnci(max_subjects=max_subjects)
    if key == config.DATASET_VIRTUAL_CURSOR["key"]:
        return load_virtual_cursor(max_subjects=max_subjects)
    if key == config.DATASET_GAZE["key"]:
        return load_gaze(max_subjects=max_subjects)
    if key == config.DATASET_ERPCORE_ERN["key"]:
        return load_erp_core_ern(max_subjects=max_subjects)
    raise ValueError(f"未知数据集 key：{key}")
