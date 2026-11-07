#!/usr/bin/env bash
# =============================================================
# ERRP 复现项目 · 一键环境配置脚本
# 用途：在全新的 Linux 沙箱（或本机）中重建可运行环境
# 用法：bash setup_env.sh
# 说明：沙箱闲置销毁后，手动安装的包会丢失，重新运行本脚本即可恢复
# =============================================================
set -euo pipefail

# 切到本脚本所在目录（项目根目录）
cd "$(dirname "$0")"

echo "[1/3] 检查 Python 版本（要求 >= 3.10）..."
python3 --version
python3 - <<'PY'
import sys
if sys.version_info < (3, 10):
        sys.exit("需要 Python 3.10 或更高版本，当前为 %s" % sys.version)
print("Python 版本检查通过")
PY

echo "[2/3] 安装/校准 Python 依赖（固定版本，见 requirements.txt）..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

echo "[3/3] 验证关键依赖可正常导入..."
python3 - <<'PY'
import numpy, scipy, sklearn, mne, matplotlib, pandas
from mne.preprocessing import Xdawn                       # 官方 xDAWN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
print("环境就绪：")
print("  numpy", numpy.__version__, "| scipy", scipy.__version__,
      "| sklearn", sklearn.__version__, "| mne", mne.__version__)
PY

echo
echo "环境配置完成。下一步："
echo "  1) 准备数据：python3 download_data.py"
echo "  2) 运行复现：python3 run_reproduction.py"
