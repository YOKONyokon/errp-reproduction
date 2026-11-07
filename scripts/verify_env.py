#!/usr/bin/env python3
"""
verify_env.py — ERRP 容器/环境自检脚本
在容器内或任意 Python 环境运行，验证：
  1. 核心科学计算栈可导入且版本符合 requirements.txt
  2. 项目自身模块（config / data_loader / ...）可导入
  3. 关键分类器与官方 xDAWN 可实例化
退出码 0 表示环境就绪，非 0 表示存在问题。
"""
import sys
from pathlib import Path

# 项目根（scripts/ 的上一级），保证能 import config 与 src 内模块
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

EXPECTED = {
    "numpy": "2.5.4",
    "scipy": "1.17.1",
    "sklearn": "1.5.2",
    "mne": "1.13.2",
    "pandas": "2.2.3",
    "matplotlib": "3.11.0",
}


def main() -> int:
    ok = True
    print("== 1. 核心依赖版本与路径 ==")
    import importlib
    for mod_name, want in EXPECTED.items():
        m = importlib.import_module(mod_name)
        path_ok = "site-packages" in (m.__file__ or "")
        flag = "OK " if m.__version__ == want and path_ok else "ERR"
        if flag == "ERR":
            ok = False
        print(f"  [{flag}] {mod_name:10s} got={m.__version__:8s} want={want:8s}")

    print("== 2. 项目模块导入 ==")
    try:
        import config
        from data_loader import load_dataset
        from features import build_feature
        from classifiers import build_classifier
        from evaluation import balanced_accuracy_score
        from discriminability import r_squared, template_snr
        print("  [OK ] config / data_loader / features / classifiers / "
              "evaluation / discriminability 全部可导入")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  [ERR] 项目模块导入失败: {exc}")

    print("== 3. 关键组件可实例化 ==")
    try:
        from mne.preprocessing import Xdawn
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.naive_bayes import GaussianNB
        from sklearn.svm import SVC
        Xdawn()
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        GaussianNB()
        SVC(kernel="rbf")
        print("  [OK ] xDAWN / sLDA / GNB / SVM 均可实例化")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  [ERR] 组件实例化失败: {exc}")

    print("=" * 46)
    print("环境自检通过，容器可运行。" if ok else "环境自检未通过，请检查上方 ERR 项。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
