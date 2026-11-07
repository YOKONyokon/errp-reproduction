# 快速复现指南（QUICKSTART）

本指南用最短路径从零复现 **6 个真实 ErrP 数据集 × 5 特征 × 4 分类器 = 120 条管线**。
完整背景与方法细节见 `README.md`；本文件只讲“怎么跑起来”。

---

## 0. 你将得到什么

| 项目 | 内容 |
|------|------|
| 数据集 | LSC、HRIcursor、HRIrobot、Coadaptation、VirtualCursor、Gaze |
| 特征 | FCzCz、Timepoints、Overlap、xDAWN、FCB（共 5 种） |
| 分类器 | GNB、sLDA、线性 SVM、RBF-SVM（共 4 种） |
| 评估 | 5 折分层交叉验证、1000 次置换检验，主指标 balanced accuracy |
| 标签口径 | 错误=1、正确=0 |

---

## 1. 系统要求

- 操作系统：Linux（已在 Ubuntu 沙箱实测）；macOS 亦可，Windows 建议用 WSL2
- Python **3.10 及以上**（实测 3.12）
- 磁盘：约 8 GB（含原始数据）；内存：4 GB 即可（xDAWN 已用稀疏矩阵防 OOM）
- 网络：能访问 GitHub 与 figshare；受限时见第 6 节

---

## 2. 三步完成复现

### 第 1 步：配置环境（二选一）

**方式 A — pip（最简单）**
```bash
bash setup_env.sh                 # 安装 requirements.txt 并自动验证导入
```

**方式 B — conda（完全隔离，可选）**
```bash
bash setup_conda.sh               # 自动装 Miniconda 并创建环境 errp
conda activate errp
```

> 有 Docker 时也可：`docker build -t errp:latest .`，详见 `Dockerfile` 顶部注释。

### 第 2 步：下载数据

```bash
python3 download_data.py
```
- 5 个数据集自动下载（已存在自动跳过、大文件断点续传）；
- **Gaze 需浏览器手动下载**：脚本会打印完整步骤（figshare 有 Cloudflare 挑战，
  命令行返回 403/202）。按提示把 12 个 .mat 放到 `third_party/gaze/` 后重跑本脚本即可。
- 只想先自动下载 5 个：`python3 download_data.py --only-auto`

下载后可选核验：
```bash
python3 scripts/verify_datasets.py   # 输出每个数据集的形状/标签/采样率卡片
python3 scripts/verify_env.py        # 仅验证环境与依赖
```

### 第 3 步：运行复现

```bash
python3 run_reproduction.py            # 全量 6 数据集，1000 次置换
```

常用变体：
```bash
python3 run_reproduction.py --quick                 # 快速冒烟（1 被试、100 次置换）
python3 run_reproduction.py --datasets LSC,Gaze     # 只跑指定数据集（逗号分隔）
python3 run_reproduction.py --hri-subjects 3        # 多被试数据集限定被试数
```

支持断点续跑：中断后重新运行会自动跳过已完成的管线。

---

## 3. 脚本一览（按使用顺序）

| 脚本 | 作用 |
|------|------|
| `setup_env.sh` / `setup_conda.sh` | 一键配置 pip / conda 环境 |
| `requirements.txt` / `environment.yml` | 固定版本依赖清单（pip / conda） |
| `Dockerfile` | 容器化构建定义 |
| `download_data.py` | 下载 6 数据集（含 Gaze 手动引导） |
| `parallel_bnci.py` | （可选）并发下载 BNCI013，未跑通集 |
| `run_reproduction.py` | 主流程：5×4 管线、交叉验证、置换检验、出图 |
| `run_optimized.py` | 只重跑 xDAWN/FCB 并重建对比（优化验证用） |
| `run_experiments.py` | 第二部分 4.2–4.5 受控实验 |
| `run_morphology.py` | 4.6 波形形态学与峰值/SNR |
| `run_regression.py` | 4.7 准确率~r²/SNR 回归 |
| `scripts/verify_env.py` / `verify_datasets.py` | 环境与数据自检 |

核心库代码全部在 `src/`：`preprocessing / data_loader / features / classifiers /
evaluation / plotting / discriminability / experiments / morphology`，每个模块顶部
均有功能说明，关键步骤逐行注释。

---

## 4. 结果产物

运行后在 `results/` 下，每个数据集生成：

- `*_bacc_matrix.csv`：5×4 balanced accuracy 数值矩阵
- `*_detailed.csv`：每条管线的 bacc、TPR、FPR、95% 机会水平、p 值
- `*_results.json`：完整结果（含混淆矩阵等）
- `*_bacc_heatmap.png`：热力图；`*_waveform.png`：两类平均波形；
  `*_confusion_*.png`：混淆矩阵

汇总文件：
- `summary_all.csv`：全部 120 条管线
- 项目根 `comparison_paper_vs_mine.csv`：与综述 Table 3 的逐格差异
- `part2_*.csv/.txt`：第二部分受控实验结果

---

## 5. 预期最佳结果（优化后实测）

| 数据集 | 最佳 bacc | 最佳管线 |
|--------|----------|----------|
| LSC | 91.6% | Overlap + sLDA |
| HRIcursor | 82.9% | Overlap + SVM |
| HRIrobot | 60.9% | Overlap + SVM |
| Coadaptation | 77.3% | FCB + RBF-SVM |
| VirtualCursor | 65.8% | Overlap + SVM |
| Gaze | 77.1% | FCB + GNB |

与论文整体差异：MAE 5.15（优化前 8.39），104 个可对比单元中 59 个差异 ≤5。

---

## 6. 网络受限提示

- GitHub 直连慢或被重置：可在 URL 前加代理前缀，如
  `https://ghfast.top/<完整GitHub URL>`，代理可用性会变，需现场测速；
- Hugging Face 镜像：设置 `HF_ENDPOINT=https://hf-mirror.com`；
- 大文件务必用 `curl --retry --retry-all-errors -C -` 断点续传；
- Gaze 只能用真实浏览器过 Cloudflare 挑战，无法用命令行绕过。

---

## 7. 复现注意事项

- 请整体使用 `requirements.txt` 的固定版本，不要单独升级 numpy/scikit-learn
  （FCB/xDAWN/MNE 对版本敏感，且新版 sklearn 会改变部分数值）；
- 所有空间滤波、归一化均在交叉验证**训练折内**拟合，无数据泄漏；
- 剩余与论文的差异来自评估协议（5 折 vs 10 折/leave-session）、基线口径与
  MATLAB/Python 工具链，属正常口径差异，不应为凑数值引入泄漏。
