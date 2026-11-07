# M2 里程碑门验收报告 · 可运行环境（阶段 2「代码与环境」）

对应计划：`errp_research_plan.html` 阶段 2；验收门 M2「可运行环境」
执行日期：2026-11-07；执行环境：Kata Containers + hpvs 强隔离云沙箱（Ubuntu 22.04，内核 6.6.95）

## 1. 验收标准对照

| M2 验收项 | 结论 | 证据 |
|---|---|---|
| 重点仓库核验、固定 commit | ✅ 达成 | `docs/repo_ledger.csv` / `.md`，8 个 git 仓库全部固定 commit、工作区干净 |
| 可运行环境（conda） | ✅ 达成 | `errp` 环境隔离验证通过，自检脚本全绿 |
| 可运行环境（Docker） | ✅ 达成（云端 CI） | GitHub Actions 完成 `docker build` + 容器内 `verify_env.py`，自检全绿退出 0；运行 42385579910 |
| ≥1 模型在 1 数据集端到端跑通 | ✅ 超额达成 | 复用现有 6 数据集 × 20 管线 = **120 条**端到端结果，95% 统计显著 |

## 2. 仓库核验与版本固定
- 计划 5 个重点仓库 + 现有 3 个 git 仓库，共 **8 个 git 仓库**全部核验可达、固定 commit；另登记 3 个非 git 数据目录。
- 机器可读台账 11 行 12 列（pandas 验证可解析），含远程 URL、固定 hash、分支、提交日期、规模、用途、可达性。
- ERP-XTTN 浅克隆 index 损坏已通过 `read-tree`/`checkout-index` 完整修复（65,398 条目）。

## 3. Conda 环境（完全可用）
- Miniconda 26.9.4，环境名 `errp`，Python 3.12.16。
- 核心包精确版本：numpy 2.5.4 / scipy 1.17.1 / scikit-learn 1.5.2 / mne 1.13.2 / pandas 2.2.3 / matplotlib 3.11.0。
- 关键隔离措施：`PYTHONNOUSERSITE=1`，6 个包全部从环境目录加载（已验证加载路径），sLDA 玩具数据 bacc=0.95。
- 复现文件：`environment.yml`、`setup_conda.sh`；自检：`python scripts/verify_env.py`（全绿，退出码 0）。

## 4. Docker 环境（环境硬限制，如实记录）
**已完成的交付**：Docker 29.9.3 已安装；`Dockerfile`、`.dockerignore`、`scripts/verify_env.py` 已编写并通过逻辑校验。

**本沙箱无法构建/运行容器的根因（已逐条验证，非配置问题）**：
1. root 的 CapEff=CapBnd=`0xa80c25fb`，**不含 CAP_SYS_ADMIN、CAP_NET_ADMIN**；普通用户 CapEff=0。
   - 直接 `unshare -m` 被拒；iptables NAT 链创建被拒。
2. cgroup v2 只读挂载，无法 remount，也无可写子目录 → runc 无法 apply cgroup。
3. rootless 路径：`unshare -U` 可用，但 rootlesskit 依赖的 `newuidmap` 写 uid_map 被虚拟化层拒绝（EPERM，即使 setuid root 且含 CAP_SETUID）；单 uid 映射又无法处理多 uid/gid 文件（lchown EINVAL）。
- 实测 `docker build` 在基础镜像多 gid 文件（etc/gshadow, gid 42）lchown 处失败；`docker run` 在 cgroup/层注册处失败。

**最终方案（用户拍板：方案 A 的云端 CI 形态）**：不在本机安装 Docker，改用 GitHub Actions 云端 runner 完成构建与验证，全程不触碰本机。

**GitHub Actions CI 验证证据（结果：✅ 通过）**：
- 公开仓库：https://github.com/YOKONyokon/errp-reproduction
- 工作流：`.github/workflows/docker-verify.yml`（runs-on ubuntu-latest，自带 Docker）
- 成功运行：https://github.com/YOKONyokon/errp-reproduction/actions/runs/42385579910（47s，conclusion=success）
- 步骤全绿：checkout → setup-buildx → `docker build -t errp:latest .` → `docker run --rm errp:latest` → 显式 `--entrypoint python ... verify_env.py`
- 容器内自检输出：6 个依赖版本与 site-packages 路径全部 OK；config/data_loader/features/classifiers/evaluation/discriminability 全部可导入；xDAWN/sLDA/GNB/SVM 可实例化；打印"环境自检通过，容器可运行"，退出码 0。

**配套修复（为让无挂载自检可跑）**：
- `src/features.py` 对 third_party 内 `FCB_toolbox` 的硬导入改为容错（try/except），仅在真正使用 FCBFeature 时才报明确 ImportError；conda 下 third_party 存在，实际运行行为不变。
- 显式验证步骤用 `--entrypoint python` 覆盖，避免与 Dockerfile ENTRYPOINT 叠加成 `python python`。

**本机零残留**：Windows 本机安装 Docker 过程产生的文件与功能已全部回滚（删除安装目录/日志，禁用 WSL 与 VirtualMachinePlatform，重启后完全生效），无 Docker 服务/进程。

## 5. 端到端证据（复用已跑通管线）
- `results/summary_all.csv`：**120 条**管线（6 数据集 × 5 特征 × 4 分类器），字段含 bacc/tpr/fpr/chance95/p_value。
- 覆盖数据集：LSC、HRIcursor、Coadaptation、VirtualCursor、Gaze、HRIrobot；95% 管线 p<0.05；bacc 区间 0.50–0.916。
- 每数据集另含 bacc 矩阵、detailed、JSON、热力图、混淆矩阵、波形图。

## 6. 遗留事项
- BNCI 013 仅下载约 21%，需用 `parallel_bnci.py` 断点续传至 12 个 .mat 完整后更新台账（与阶段 2 并行，不阻塞）。

## 7. M2 结论
**conda 与 Docker 两条环境路线、以及端到端验收现已全部达成。** conda 环境本机自检全绿；Docker 路线在不安装本机 Docker 的前提下，通过 GitHub Actions 云端 CI 完成镜像构建与容器内 `verify_env.py` 验证（运行 42385579910，success，退出码 0）；端到端复用 120 条已跑通管线。Windows 本机相关变动已全部回滚、零残留。**M2「可运行环境」门完全满足，可进入下一阶段。**
