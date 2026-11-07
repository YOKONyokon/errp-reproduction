# 阶段 3 数据与预处理 — 数据 Pipeline 说明

> 对应计划 HTML 阶段 3「数据与预处理」。本阶段无独立里程碑门（M3 在阶段 4 末），
> 交付物为可复现的数据 pipeline：加载器 + 预处理 + 划分协议 + 数据卡片。
> 标签约定统一为 **0=正确、1=错误**。

## 1. Pipeline 总览

```
原始数据（.mat / EEGLAB .set+.fdt / numpy）
        │  src/data_loader.py
        ├─ 读取 → 剔除 EOG →（连续数据）1–10Hz 零相位带通
        ├─ 按事件切 epoch（反馈锁定 / 反应锁定）→ 基线校正
        ├─（ERP CORE）1024→256 Hz 降采样
        ▼
EpochData  X(n,ch,t) · y · ch_names · sfreq · subjects · tmin
        │  src/splits.py
        ├─ within-session 分层 5 折   （论文主口径）
        ├─ LOSO 留一被试              （跨被试泛化）
        └─ grouped 按组留一           （cross-task）
        ▼
特征提取（折内 fit，防泄漏）→ 分类器
```

## 2. 数据集清单与数据卡片

本阶段共注册 **8 个数据集**（7 个反馈/任务数据 + 1 个反应锁定 ERP CORE）。
完整机器可读卡片见 [`docs/dataset_cards.json`](dataset_cards.json)，由
`scripts/verify_datasets.py` 自动生成。

| key | 试次 | 错误率 | 通道 | 采样率 | 时间点 | 窗口 (s) | 被试 | 单位 | 状态 |
|---|---:|---:|---:|---:|---:|---|---:|---|---|
| LSC | 190 | .147 | 12 | 256 | 256 | [−.2, .796] | 1 | µV | 已核验 |
| HRIcursor | 5317 | .340 | 27 | 256 | 257 | [−.2, .8] | 11 | V | 已核验 |
| HRIrobot | 5822 | .353 | 27 | 256 | 257 | [−.2, .8] | 12 | V | 已核验 |
| Coadaptation | 3900 | .396 | 27 | 256 | 257 | [−.2, .8] | 13 | V | 已核验 |
| BNCI013 | — | — | 64 | 512 | — | 反馈锁定 | 6 | V | 后台续传 |
| VirtualCursor | 4230 | .466 | 16 | 256 | 257 | [−.2, .8] | 8 | V | 已核验 |
| Gaze | 4931 | .104 | 61 | 256 | 180 | [−.2, .499] | 10 | µV | 已核验 |
| ERPCORE_ERN | — | — | 30 | 1024→256 | — | [−.2, .6] 反应锁定 | 40 | V | 后台下载 |

核验结论：6 个已加载数据集**全部无 NaN**、dtype 均为 float64，标签分布、
通道数、采样率、窗口与 `config.py` 注册表一致。

## 3. 加载器架构（`src/data_loader.py`）

- 统一容器 `EpochData`：`X (n_trials, n_channels, n_times)`、`y`、`ch_names`、
  `sfreq`、`subjects`、`tmin`。
- 每个数据集一个加载函数，统一入口 `load_dataset(key, max_subjects=None)`，
  共 8 个分支：LSC、HRIcursor、HRIrobot、Coadaptation、BNCI013、
  VirtualCursor、Gaze、ERPCORE_ERN。
- 计划反复强调 MOABB 不支持 ErrP，因此全部为**自定义 MNE 加载器**，未依赖 MOABB。
- 事件映射（部分）：HRI 用 `S 4`(正确)/`S 6`(错误)；Coadaptation 用 33031/33033；
  BNCI 用 5/10(正确)、6/9(错误)；ERP CORE 用反应三位码末位 1/2 判对错。

## 4. 预处理口径（`src/preprocessing.py` + 加载器内 MNE 调用）

按数据初始状态分两条执行路径：

1. **连续 raw 路径**（HRI/Coadaptation/BNCI/VirtualCursor/ERPCORE）：
   剔除 EOG → **1–10 Hz 带通**（MNE FIR、零相位）→ 切 epoch →
   基线校正（[−.2, 0]）；ERP CORE 额外降采样到 256 Hz。
2. **已切 epoch 路径**（LSC/Gaze）：数据发布前已滤波切好，加载时仅做
   基线校正，不重复滤波以避免 epoch 边界二次伪影。

- 共平均参考（CAR）已实现但**论文未统一使用**，主 pipeline 未调用，仅作可选工具。
- `preprocessing.py` 另提供 scipy 版 Butterworth（filtfilt 零相位）用于原理对照。

### 单位口径（已知现象）

不同加载器输出单位不统一：**LSC、Gaze 为 µV；其余经 MNE 的数据集为 V**。
- 对分类结果影响很小：各数据集独立评估，且收缩 LDA / 带标准化的分类器对
  整体单调缩放不敏感。
- 分析层已容错：`src/morphology.py::to_microvolt()` 按最大绝对值（阈值 .01）
  自动把 V 换算为 µV，`src/plotting.py` 同样处理，保证波形幅值与论文 µV 口径对齐。
- 首次核验时 V 数据集（幅度约 1e-3）经 `round(.,2)` 显示为 0 属显示精度问题，
  非数据全零；核验脚本已改为科学计数法并标注单位。

## 5. 划分协议（`src/splits.py`）

| 协议 | 函数 | 分组依据 | 评估含义 |
|---|---|---|---|
| within-session | `within_session_splits` | 无（按标签分层） | 同一人群内对新试次分类，论文主口径 |
| LOSO | `loso_splits` | 被试标识 | 跨被试泛化，测试被试不进入训练 |
| cross-task | `grouped_splits` | 任务/组标识 | 跨任务留一，按组隔离 |

### 防泄漏验证（已实测通过）

- 现有 `evaluate_pipeline` 中**特征提取器（xDAWN/FCB）与分类器均在每折训练数据上
  fit**，再 transform 验证折，避免特征级泄漏。
- `splits.py` 提供两级泄漏检查并已验证：
  - within-session：5 折索引无重叠、并集覆盖全部、各折错误率贴近总体（.147）。
  - LOSO：测试被试与训练被试完全不相交（`assert_group_disjoint` 全 True）。

### 端到端验证（Overlap + sLDA，已实测）

- within-session 冒烟（`scripts/smoke_pipeline.py`）：6 个数据集全部跑通且非 chance，
  bacc 依次为 LSC .916、HRIcursor .792、HRIrobot .580、Coadaptation .772、
  VirtualCursor .646、Gaze .684，与历史最佳趋势一致。
- LOSO 跨被试（`scripts/verify_loso.py`，VirtualCursor 8 被试）：逐被试
  bacc .505–.691，**均值 .619**（std .067），低于 within-session 符合跨被试更难的预期。

## 6. 待办与依赖（数据下载，后台进行）

- **BNCI013**：`parallel_bnci.py` 2 worker 断点续传，完成后补卡片并更新台账。
- **ERPCORE_ERN（NEMAR nm000132）**：`parallel_nemar.py` 2 worker 仅下载
  flankers 任务（避免全量 17.5GB）；脚本已修复限流时 HEAD 返回错误页的问题，
  加入大小合理性校验与退避重试。
- 下载完成后需用 MNE 实际加载做最终完整性校验，并回填本文件第 2 节占位行。
