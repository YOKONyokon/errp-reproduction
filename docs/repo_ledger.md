# 阶段 2 · 仓库核验台账（M2 门可追溯文档）

- 核验日期：2026-11-07
- 核验范围：最初计划 HTML 列出的 5 个重点仓库 + 当前复现项目实际使用的仓库/数据目录，共 **12 项**（8 个 git 仓库 + 4 个非 git 数据目录；阶段 3 新增 NEMAR nm000132）
- 核验方式：`git ls-remote` 确认远程可达 → 克隆到 `third_party/` → 记录远程 URL、固定 commit hash、分支、提交日期、语言、依赖文件与用途
- 机器可读版本：[`repo_ledger.csv`](./repo_ledger.csv)

---

## 一、计划重点仓库（5 个）

### 1. RL-BCI-Lab/deepbci — 通用生物传感深度学习框架

| 项 | 值 |
|----|----|
| 远程 URL | https://github.com/RL-BCI-Lab/deepbci.git |
| 固定 commit | `9a0f59b7a28a3c2f26003ba73ceff9990e4f7071` |
| 分支 / 日期 | master / 2022-12-05 |
| 语言 / 规模 | Python（86 .py、6 .ipynb）/ 56M |
| 依赖/环境 | `setup.py`、`docker/environment.yaml`（conda，python 3.9）、`docker/Dockerfile` |
| 用途 | 面向 EEG 的深度网络与深度强化学习框架，统一多数据集格式；当前 HEAD 即 CIBCI 2022 ErrP 变异论文配套代码，含 agents/models/data 等模块 |
| 核验结论 | 远程可达、克隆成功、自带 conda 与 Docker 方案 |

### 2. DL4mHealth/ERP-Benchmark — 2026 IEEE TBME ERP 基准

| 项 | 值 |
|----|----|
| 远程 URL | https://github.com/DL4mHealth/ERP-Benchmark.git |
| 固定 commit | `e2d9c772e3406aea10e98f22042943b58612dd28` |
| 分支 / 日期 | main / 2026-09-29 |
| 语言 / 规模 | Python（60 .py、10 .ipynb）/ 8.3M |
| 依赖/环境 | `requirements.txt`、`meta_run_*.sh` |
| 用途 | 12 个 ERP 数据集 × 15 种方法（2 手工特征 + 10 深度学习 + 3 基础模型），任务含 ERP 刺激分类与脑疾病检测；提供统一预处理与训练 pipeline |
| 核验结论 | 远程可达、克隆成功；注意其口径为泛指 ERP，非仅 ErrP，可作为方法库参考 |

### 3. cgenevier/ERP-XTTN — 可解释交叉注意力 ERP 分类器

| 项 | 值 |
|----|----|
| 远程 URL | https://github.com/cgenevier/ERP-XTTN.git |
| 固定 commit | `e25de7955b3198bb193ea61cf75922efcf91e414` |
| 分支 / 日期 | main / 2026-08-28 |
| 语言 / 规模 | Python（18 .py、6 .m）+ 63,598 个 .npz 结果文件 / 22G |
| 依赖/环境 | `requirements.txt`（torch 2.8、mne 1.12、pyriemann、scikit-learn 1.8） |
| 用途 | 神经生理学约束、原型引导的跨被试 ERP 分类器（峰值单元 + 交叉注意力 + 振幅通路晚期融合）；含 BNCI ErrP 013 数据集及 EEGNet、EEG-Deformer、EPMN、xDAWN+黎曼基线 |
| 核验结论 | 远程可达；浅克隆时曾出现 index 缺失（工作区文件被误判 deleted），已用 `git read-tree HEAD` + `git checkout-index -a -f` 完整恢复，当前工作区干净（65,398 文件全检出） |

### 4. lucklab/ERP_CORE — ERP 开放资源（含 ERN）

| 项 | 值 |
|----|----|
| 远程 URL | https://github.com/lucklab/ERP_CORE |
| 固定 commit | `c18b43d70d791ca914d90410afe4ff06d6f7f429` |
| 分支 / 日期 | master / 2020-08-13 |
| 语言 / 规模 | MATLAB（152 .m）/ 491M |
| 依赖/环境 | MATLAB + EEGLAB（各成分目录内含脚本） |
| 用途 | 7 个 ERP 成分（6 范式）、40 被试；其中 **ERN（Flankers 范式）** 与本项目直接相关，含 40 被试处理结果、19 个 .m 分析脚本；原始 EEG 需另从 OSF（osf.io/thsqg）下载 |
| 核验结论 | 远程可达、克隆成功；GitHub 内为分析脚本与中间结果，非原始数据 |

### 5. stefan-ehrlich/dataset-ErrP-HRI — 人机交互 ErrP 数据

| 项 | 值 |
|----|----|
| 远程 URL | https://github.com/stefan-ehrlich/dataset-ErrP-HRI.git |
| 固定 commit | `4bc3e52469934aa0551a2261a6e4e899d156c28f` |
| 分支 / 日期 | master / 2026-01-22 |
| 语言 / 规模 | EEGLAB 数据（24 .set + 24 .fdt）/ 2.4G |
| 用途 | 屏幕光标（cursor）与 NAO 机器人（robot）两类监控 ErrP，27 通道 256 Hz；本项目 HRIcursor/HRIrobot 两数据集即来源于此 |
| 核验结论 | 远程可达、已接入主管线 |

---

## 二、现有项目仓库（3 个 git 仓库）

### 6. gpiresML/FCB-spatial-filter — FCB 空间滤波官方工具箱

- 固定 commit：`0b91fbd3ef9ec4bf2fe79f14c10b387b0e97bae6`（main，2024-02-22）
- 语言/规模：Python 3 + MATLAB 5 + 2 ipynb，116M
- 用途：FCB（Fisher 准则）空间滤波官方 Python/MATLAB 工具箱，内含 LSC 拼写器数据；本项目 FCB 特征与 LSC 数据集来源
- 核验结论：已接入；工作区有 1 个运行时生成的未跟踪文件 `__pycache__/*.pyc`（非实质变更，不影响版本固定）

### 7. stefan-ehrlich/dataset-ErrP-coadaptation — 人机协同自适应 ErrP 数据

- 固定 commit：`d7156e7f4f475949871e28012ca5254d626483b0`（master，2026-01-22）
- 语言/规模：EEGLAB 数据（82 .set + 82 .fdt）+ 日志，3.2G
- 用途：13 被试、27 通道、256 Hz，含 calib/corl1/corl2/corl4 多阶段；本项目 Coadaptation 数据集来源
- 核验结论：已接入主管线

### 8. flowersteam/self_calibration_BCI_plosOne_2015 — 自校准虚拟光标

- 固定 commit：`68515d07186bf6956c31b98c8307c077b37b018`（master，2015-07-08）
- 语言/规模：MATLAB 24 + 8 .mat，186M
- 用途：Iturrate 等 2015 自校准 BCI 代码仓库；主仓库不含原始 EEG，原始数据在其 GitHub Release
- 核验结论：代码已接入参考

---

## 三、非 git 数据目录（4 个，来源与版本固定）

| 目录 | 来源 | 版本固定方式 | 规模 | 状态 |
|------|------|--------------|------|------|
| `bnci-013/` | TU Graz 镜像 https://lampx.tugraz.at/~bci/database/013-2015/ | 记录逐文件 URL + 下载脚本 `parallel_bnci.py` | 825M | **后台断点续传，约 25%**；Subject01 完整，Subject02 及后续续传中 |
| `gaze/` | figshare article 5938714，DOI `10.6084/m9.figshare.5938714.v1` | DOI 固定版本 + 10 被试文件清单 | 403M | 已完整下载并接入（10 被试、61 通道、4931 试次） |
| `virtual-cursor-release/` | GitHub Release `.../releases/download/plosOne/data_1.zip` | 固定 Release tag（plosOne）+ zip 文件名 | 1.1G | 已下载解压并接入（8 被试、16 通道） |
| `nm000132/` | NEMAR https://nemar.org/dataset/nm000132 | 记录逐文件 URL + 下载脚本 `parallel_nemar.py`，仅下 flankers | 17M | **后台 2 worker 下载中**；ERP CORE 反应锁定 ERN，30 通道 1024Hz |

---

## 四、固定版本与可复现约定

1. **git 仓库**：全部以完整 commit hash（非 tag、非分支名）固定，上表 8 个 hash 即复现基准；后续 `git pull` 不改变已记录基准，如需升级须重新登记。
2. **非 git 数据**：以 DOI / 不可变 Release tag / 完整直链 URL 固定，并保留下载脚本与文件清单。
3. **异常处理留痕**：ERP-XTTN 的浅克隆 index 缺失问题及修复命令已记录，避免重复踩坑。
4. **待办**：BNCI 013 与 NEMAR flankers 后台续传，完成后更新本台账状态列并回填数据卡片。

---

## 五、阶段 3 补充：数据 pipeline 新增模块

- `src/data_loader.py`：新增 `load_erp_core_ern()` 反应锁定加载器并注册到统一入口（现 8 个数据集）。
- `src/splits.py`（新增）：固化 within-session 分层 5 折、LOSO、按组留一（cross-task）划分，
  并提供样本级/组级泄漏检查；已实测验证无重叠、LOSO 被试完全隔离。
- `scripts/verify_datasets.py`（新增）：自动生成数据卡片 `docs/dataset_cards.json`（6 数据集无 NaN）。
- `parallel_nemar.py`（新增）：2 worker 断点续传 NEMAR flankers，含 HEAD 合理性校验与退避重试。
- 完整 pipeline 说明见 [`docs/data_pipeline.md`](./data_pipeline.md)。
