#!/usr/bin/env bash
# =============================================================
# ERRP 复现项目 · Conda 环境一键配置脚本（阶段 2 交付物）
# 用途：在全新 Linux x86_64 机器上重建隔离的 conda 环境 "errp"
# 用法：bash setup_conda.sh
# 复现：也可直接  conda env create -f environment.yml
# 关键：禁用用户级 site-packages（PYTHONNOUSERSITE），保证依赖全部
#       装入 conda 环境而非 ~/.local，环境可独立打包进 Docker
# =============================================================
set -euo pipefail
cd "$(dirname "$0")"

ENV_NAME="errp"
CONDA_ROOT="${HOME}/miniconda3"

# ---- [1/4] 如无 conda，则安装 Miniconda ----
if [ ! -x "${CONDA_ROOT}/bin/conda" ]; then
  echo "[1/4] 未检测到 conda，下载并静默安装 Miniconda ..."
  INSTALLER="$(mktemp --suffix=.sh)"
  curl -fsSL -o "$INSTALLER" \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash "$INSTALLER" -b -p "$CONDA_ROOT"
  rm -f "$INSTALLER"
else
  echo "[1/4] 检测到 conda：$CONDA_ROOT"
fi

# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

# Anaconda 新渠道需接受服务条款（已接受则忽略错误）
conda tos accept --override-channels \
  --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels \
  --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# ---- [2/4] 创建环境（已存在则跳过）----
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[2/4] 环境 $ENV_NAME 已存在，跳过创建"
else
  echo "[2/4] 创建环境 $ENV_NAME (Python 3.12) ..."
  conda create -y -n "$ENV_NAME" python=3.12
fi
conda activate "$ENV_NAME"

# ---- [3/4] 隔离用户级包并安装固定版本依赖 ----
echo "[3/4] 安装固定版本依赖（requirements.txt），强制装入本环境 ..."
export PYTHONNOUSERSITE=1
python -m pip install --upgrade pip
python -m pip install --force-reinstall -r requirements.txt

# ---- [4/4] 验证导入路径确实在 conda 环境内 ----
echo "[4/4] 验证关键依赖与导入路径 ..."
python - <<'PY'
import numpy, scipy, sklearn, mne, pandas, matplotlib, sys
mods = (numpy, scipy, sklearn, mne, pandas, matplotlib)
for m in mods:
    assert "miniconda3/envs/errp" in m.__file__, (m.__name__, m.__file__)
    print(f"  {m.__name__:10s} {m.__version__:8s} {m.__file__}")
from mne.preprocessing import Xdawn
print("Conda 环境隔离验证通过，所有包均来自 errp 环境目录")
PY

echo
echo "Conda 环境配置完成。激活方式：conda activate $ENV_NAME"
echo "导出复现文件：environment.yml（conda env create -f environment.yml）"
