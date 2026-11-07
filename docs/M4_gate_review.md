# M4 里程碑门验收 — 前沿方法与迁移（阶段 5）

> 对应计划 HTML 阶段 5「前沿与迁移」，里程碑门 **M4**。
> 范围：在**已下载的 6 个真实数据集**上完成深度基线、跨任务迁移与小样本学习曲线。
> BNCI 013、NEMAR 因下载未完成暂不纳入（为释放 CPU 已暂停，断点可续传）。
> 主指标：balanced accuracy；标签 0=正确、1=错误。

## 1. 验收结论（一句话）

**M4 通过。** 完成 EEGNet 深度基线（6 集）、跨任务迁移（pairwise + 留一数据集，
含欧氏对齐子空间正则）与小样本学习曲线（经典 vs 深度），三类证据齐备，
满足 M4 标准"跨任务/小样本迁移实验完成、学习曲线与前沿方法对比齐备"。

## 2. 深度基线：EEGNet（6 集 within-session）

独立实现标准 EEGNet（Lawhern 2018），CPU 训练、30 epoch、balanced CE、早停。

| 数据集 | LSC | HRIcursor | HRIrobot | Coadaptation | VirtualCursor | Gaze |
|---|---:|---:|---:|---:|---:|---:|
| EEGNet bacc | .652 | .836 | .592 | .782 | .635 | **.849** |
| 经典最佳（M3） | .916 | .829 | .609 | .773 | .658 | .771 |

- 大数据集（Gaze、HRIcursor、Coadaptation）上 EEGNet 与经典方法相当或更优，
  Gaze 上 .849 明显高于经典最佳 .771；小样本 LSC（190 试次）上深度模型欠拟合。
- 网络极轻量（27ch 仅约 1.8k 参数），瓶颈在 CPU 训练时长（单集 3–11 分钟）。

## 3. 跨任务迁移（27ch 同构三集）

三个 27 通道数据集（HRIcursor/HRIrobot/Coadaptation）通道完全同构，信息零损失。
模型：Overlap + shrinkage sLDA；对比 raw 与 EA（欧氏对齐）。

### 3.1 within-session 参考
EA 在同数据集内**稳定小幅提升**：HRIcursor .792→.805、HRIrobot .580→.588、
Coadaptation .772→.774。

### 3.2 pairwise 迁移（train A → test B，bacc）

| 源 → 目标 | raw | EA |
|---|---:|---:|
| HRIcursor → HRIrobot | .540 | .548 |
| HRIcursor → Coadaptation | .530 | .511 |
| HRIrobot → HRIcursor | **.645** | .626 |
| HRIrobot → Coadaptation | .451 | .472 |
| Coadaptation → HRIcursor | .523 | .506 |
| Coadaptation → HRIrobot | .500 | .499 |

### 3.3 留一数据集 LODO（合并另两个训练 → 测试第三个）

| 目标 | HRIcursor | HRIrobot | Coadaptation |
|---|---:|---:|---:|
| raw | .607 | .524 | .505 |
| EA | .597 | .528 | .491 |

### 3.4 迁移结论（含诚实的负面结果）

- **跨任务显著难于 within**：迁移 bacc 多在 .45–.65，明显低于 within（.58–.79），
  证明域偏移真实存在；但多数仍高于 chance（.5），说明 ErrP 特征有跨数据集通用性。
- **合并多源更稳**：LODO 整体优于单源 pairwise。
- **EA 子空间对齐效果不稳定**：within 内稳定小幅提升，但跨任务上有升有降，
  无一致增益。这是诚实的混合结果——三集虽通道同构，但范式/被试差异较大，
  仅靠二阶协方差对齐不足以消除分布偏移。

## 4. 小样本学习曲线

固定 20% 测试集，训练池按 5/10/25/50/100% 分层抽样，每点 3 种子。

### 4.1 经典 sLDA（6 集 bacc）

| 数据集 | 5% | 10% | 25% | 50% | 100% |
|---|---:|---:|---:|---:|---:|
| HRIcursor | .702 | .752 | .772 | .796 | .810 |
| Coadaptation | .692 | .704 | .734 | .752 | .762 |
| Gaze | .554 | .592 | .631 | .650 | .651 |
| HRIrobot | .538 | .568 | .566 | .578 | .590 |
| VirtualCursor | .578 | .599 | .625 | .647 | .647 |
| LSC | .764 | .797 | .729 | .800 | .917 |

### 4.2 经典 vs 深度（Coadaptation）

| 训练比例 | 5% | 10% | 25% | 50% | 100% |
|---|---:|---:|---:|---:|---:|
| sLDA | .692 | .704 | .734 | .752 | .762 |
| EEGNet | .631 | .639 | .686 | .736 | .758 |

- **小样本下 sLDA 明显更优**（5% 时 .692 vs .631）：shrinkage 正则在小样本下更稳健。
- 随样本量增加 EEGNet 快速追赶，**100% 时两者持平**（.758 vs .762）。
- 印证深度模型数据效率低、需要更多样本，而经典收缩方法在小样本场景更有优势。

## 5. 可复现性

- 环境：conda `errp`（Python 3.12，torch 2.5.1+cpu）；种子 `config.RANDOM_SEED=42`。
- 复现命令：
  - 深度基线：`python run_deep_baseline.py`
  - 迁移：`python run_transfer.py`
  - 学习曲线：`python run_learning_curve.py --model sLDA`（EEGNet 加 `--model EEGNet`）
- 机器可读结果：`results/{deep_baseline,transfer,learning_curve_*}.{json,csv}`。

## 6. 门结论与后续

- **M4 通过**：深度基线、跨任务迁移、小样本曲线三类证据完整，结论含正面与诚实负面结果。
- 待 BNCI/NEMAR 下载完成后补充对应实验，不改变当前门结论。
- 下一阶段：阶段 6「评估与分析」（CD 图、SHAP、波形叠加、误差分层）。
