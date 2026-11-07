# -*- coding: utf-8 -*-
"""
download_data.py
================
自动准备复现所需的 6 个可获取 ErrP 数据集（均免登录、可稳定获取）。

数据获取通道（按可靠性选择）：
    1) Git 浅克隆：FCB、HRI(cursor+robot)、Coadaptation、Virtual Cursor 主仓库
    2) GitHub Release 下载并解压：Virtual Cursor 的原始 EEG（data_1.zip）
    3) TU Graz lampx 直链：BNCI 013-2015（主源 bnci 官网不可达，改用镜像）

脚本特性：
    - 幂等（idempotent）：目标文件/目录已存在时自动跳过，可重复运行；
    - 大文件用 curl 断点续传（-C -）与自动重试，适配不稳定网络；
    - 每个数据集都打印来源链接与许可，便于核对。

仍无法从本环境获取的 5 个数据集（需登录或服务器封锁）在 README 中如实说明：
    Kaggle、Game Agent、Gaze Speller、TactileV、TactileVT。

用法：
    python3 download_data.py
"""

import subprocess
import sys
import zipfile
from pathlib import Path

import config

# ============================================================
# 1) Git 浅克隆仓库定义
# ============================================================
REPOS = [
    {
        "target": config.THIRD_PARTY_DIR / "FCB-spatial-filter",
        "url": "https://github.com/gpiresML/FCB-spatial-filter.git",
        "title": "FCB 官方 Python 工具箱（含 LSC ErrP 数据）",
        "license": "MIT",
        "links": [
            "LSC 数据：Python/data/ErrP_LSC_dataset.mat",
            "来源：https://github.com/gpiresML/FCB-spatial-filter",
        ],
    },
    {
        "target": config.HRI_DIR,
        "url": "https://github.com/stefan-ehrlich/dataset-ErrP-HRI.git",
        "title": "HRI cursor/robot 数据集（Ehrlich & Cheng 2018/2019）",
        "license": "见数据集文档",
        "links": [
            "cursor：cursor/s02_cursor.set ...",
            "robot：robot/s02_robot.set ...",
            "来源：https://github.com/stefan-ehrlich/dataset-ErrP-HRI",
        ],
    },
    {
        "target": config.COADAPTATION_DIR,
        "url": "https://github.com/stefan-ehrlich/dataset-ErrP-coadaptation.git",
        "title": "Coadaptation 数据集（Ehrlich & Cheng 2018）",
        "license": "见数据集文档",
        "links": [
            "data_coadaptation：s03–s18 的 calib/corl .set",
            "data_cursor：s01–s18 的 cursor .set",
            "来源：https://github.com/stefan-ehrlich/dataset-ErrP-coadaptation",
        ],
    },
    {
        "target": config.THIRD_PARTY_DIR / "virtual-cursor",
        "url": "https://github.com/flowersteam/self_calibration_BCI_plosOne_2015.git",
        "title": "Virtual Cursor 主仓库（代码与 expData 参考）",
        "license": "见仓库 LICENSE",
        "links": [
            "注意：原始 EEG 不在主仓库，需另外下载 Release data_1.zip",
            "来源：https://github.com/flowersteam/self_calibration_BCI_plosOne_2015",
        ],
    },
]

# Virtual Cursor 原始 EEG 的 Release 地址（data_1.zip 含 data/raw_data/EEG_s*.mat）
VIRTUAL_CURSOR_ZIP_URL = (
    "https://github.com/flowersteam/self_calibration_BCI_plosOne_2015/"
    "releases/download/plosOne/data_1.zip"
)
VIRTUAL_CURSOR_ZIP_PATH = config.THIRD_PARTY_DIR / "virtual-cursor-release" / "data_1.zip"

# BNCI 013-2015 的 TU Graz 镜像基址
BNCI_BASE_URL = "https://lampx.tugraz.at/~bci/database/013-2015/"


def run_git_clone(url: str, target: Path):
    """执行 git 浅克隆；target 已存在则跳过。"""
    if target.exists():
        print(f"  - 已存在，跳过：{target.name}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1", url, str(target)]
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _remote_size(url: str):
    """通过 HEAD 请求获取远端文件的期望字节数（失败返回 None）。"""
    res = subprocess.run(["curl", "-sIL", url], capture_output=True, text=True)
    for line in res.stdout.splitlines():
        if line.lower().startswith("content-length:"):
            return int(line.split(":", 1)[1].strip())
    return None


def download_with_retry(url: str, target: Path):
    """
    用 curl 断点续传 + 自动重试下载大文件。
    只有本地大小与远端期望大小一致时才跳过，避免把残缺文件误判为完整。
    """
    if target.exists():
        local_size = target.stat().st_size
        expected = _remote_size(url)
        if expected is not None and local_size == expected:
            print(f"  - 已完整，跳过：{target.name}")
            return
        if expected is not None:
            print(f"  - 残缺({local_size}/{expected})，续传：{target.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    # -C - 断点续传；--retry 失败自动重试；--retry-all-errors 覆盖连接中断
    cmd = [
        "curl", "-sL", "--retry", "8", "--retry-all-errors",
        "-C", "-", url, "-o", str(target),
    ]
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def prepare_virtual_cursor():
    """下载并解压 Virtual Cursor 原始 EEG（data_1.zip）。"""
    print("\n[5] Virtual Cursor 原始 EEG（GitHub Release data_1.zip）")
    # 若 raw_data 已含 8 个 EEG mat，则视为已完成
    raw_dir = config.VIRTUAL_CURSOR_DIR
    existing = list(raw_dir.glob("EEG_s*.mat")) if raw_dir.exists() else []
    if len(existing) >= 8:
        print("  - 8 个 EEG_s*.mat 已就位，跳过下载")
        return
    download_with_retry(VIRTUAL_CURSOR_ZIP_URL, VIRTUAL_CURSOR_ZIP_PATH)
    print("  - 解压 data_1.zip ...")
    with zipfile.ZipFile(VIRTUAL_CURSOR_ZIP_PATH) as zf:
        zf.extractall(config.THIRD_PARTY_DIR / "virtual-cursor-release")
    print("  - 来源：", config.DATASET_VIRTUAL_CURSOR["source_url"])


def prepare_bnci():
    """从 TU Graz 镜像下载 BNCI 013-2015 的 12 个 .mat（6 被试 × 2 run）。"""
    print("\n[6] BNCI 013-2015（TU Graz lampx 镜像直链）")
    out_dir = config.BNCI_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in config.DATASET_BNCI["subjects"]:
        for run in config.DATASET_BNCI["runs"]:
            filename = f"Subject{sub:02d}_{run}.mat"
            url = BNCI_BASE_URL + filename
            download_with_retry(url, out_dir / filename)
    print("  - 来源：", config.DATASET_BNCI["source_url"])


def verify_dataset_files():
    """下载后核对关键数据文件存在，否则提前报错（fail-fast）。"""
    checks = [
        config.DATASET_LSC["mat_path"],
        config.DATASET_HRI["data_dir"],
        config.DATASET_HRI_ROBOT["data_dir"],
        config.DATASET_COADAPTATION["data_dir"],
        config.BNCI_DIR,
        config.VIRTUAL_CURSOR_DIR,
    ]
    missing = [p for p in checks if not Path(p).exists()]
    if missing:
        sys.exit("数据准备失败，缺少：\n  " + "\n  ".join(map(str, missing)))
    print("\n[校验] 6 个数据集的关键文件均已就位。")


def main():
    print("=" * 64)
    print("ERRP 复现 · 数据准备（6 个可获取数据集，免登录）")
    print("=" * 64)
    config.THIRD_PARTY_DIR.mkdir(parents=True, exist_ok=True)

    # 1–4：git 克隆
    for i, repo in enumerate(REPOS, start=1):
        print(f"\n[{i}] {repo['title']}  [许可：{repo['license']}]")
        run_git_clone(repo["url"], repo["target"])
        for link in repo["links"]:
            print("    *", link)

    # 5：Virtual Cursor 原始 EEG
    prepare_virtual_cursor()

    # 6：BNCI 直链
    prepare_bnci()

    verify_dataset_files()
    print("\n数据准备完成，可运行：python3 run_reproduction.py")


if __name__ == "__main__":
    main()
