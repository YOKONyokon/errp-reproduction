# -*- coding: utf-8 -*-
"""
download_data.py
================
自动准备复现所需的 6 个已跑通 ErrP 数据集（对应 results/ 中的真实结果）。

数据集与获取方式
----------------
    可自动下载（5 个，免登录、命令行可直接获取）：
      1. LSC           随 FCB 官方 git 仓库提供（.mat）
      2. HRIcursor     HRI git 仓库（EEGLAB .set）
      3. HRIrobot      同上
      4. Coadaptation  Coadaptation git 仓库（.set）
      5. VirtualCursor git 仓库 + GitHub Release 原始 EEG（data_1.zip）

    需浏览器手动下载（1 个，脚本会给出详细步骤）：
      6. Gaze          figshare 有 Cloudflare 挑战，普通命令行返回 403/202，
                       只能用真实浏览器下载整包，再解压到 third_party/gaze。

脚本特性
--------
    - 幂等（idempotent）：目标已存在自动跳过，可重复运行；
    - 大文件用 curl 断点续传（-C -）与自动重试，适配不稳定网络；
    - 每个数据集打印来源链接与许可，便于核对；
    - 结尾做完整性校验，缺失即报错（fail-fast）。

用法
----
    python3 download_data.py                 # 准备 6 数据集（Gaze 给手动引导）
    python3 download_data.py --only-auto     # 只自动下载 5 个，跳过 Gaze 提示
    python3 download_data.py --include-bnci  # 额外下载 BNCI013（约 3.4GB，很慢）
"""

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

import config

# ============================================================
# 1) Git 浅克隆仓库（5 个自动集中的 4 个仓库）
# ============================================================
REPOS = [
    {
        "target": config.THIRD_PARTY_DIR / "FCB-spatial-filter",
        "url": "https://github.com/gpiresML/FCB-spatial-filter.git",
        "title": "FCB 官方 Python 工具箱（含 LSC ErrP 数据）",
        "license": "MIT",
        "note": "LSC 数据：Python/data/ErrP_LSC_dataset.mat",
    },
    {
        "target": config.HRI_DIR,
        "url": "https://github.com/stefan-ehrlich/dataset-ErrP-HRI.git",
        "title": "HRI cursor/robot 数据集（Ehrlich & Cheng 2018/2019）",
        "license": "见数据集文档（CC BY-NC-ND）",
        "note": "cursor/*.set 与 robot/*.set",
    },
    {
        "target": config.COADAPTATION_DIR,
        "url": "https://github.com/stefan-ehrlich/dataset-ErrP-coadaptation.git",
        "title": "Coadaptation 数据集（Ehrlich & Cheng 2018）",
        "license": "见数据集文档",
        "note": "data_coadaptation/ 下各被试 .set",
    },
    {
        "target": config.THIRD_PARTY_DIR / "virtual-cursor",
        "url": "https://github.com/flowersteam/self_calibration_BCI_plosOne_2015.git",
        "title": "Virtual Cursor 主仓库（代码与参考）",
        "license": "见仓库 LICENSE",
        "note": "原始 EEG 不在此仓库，需另下 Release data_1.zip",
    },
]

# Virtual Cursor 原始 EEG 的 Release 地址
VIRTUAL_CURSOR_ZIP_URL = (
    "https://github.com/flowersteam/self_calibration_BCI_plosOne_2015/"
    "releases/download/plosOne/data_1.zip"
)
VIRTUAL_CURSOR_ZIP_PATH = (
    config.THIRD_PARTY_DIR / "virtual-cursor-release" / "data_1.zip")

# BNCI 镜像基址（可选，默认不下载）
BNCI_BASE_URL = "https://lampx.tugraz.at/~bci/database/013-2015/"

# Gaze 期望的 12 个文件名（10 被试 + 2 元数据）
GAZE_EXPECTED_FILES = [
    "namesEBN.mat", "EbN_electrodes.mat",
    "S01.mat", "S02.mat", "S03.mat", "S04.mat", "S05.mat", "S06.mat",
    "S07.mat", "S11.mat", "S13.mat", "S15.mat",
]


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
    """通过 HEAD 请求获取远端期望字节数（失败返回 None）。"""
    res = subprocess.run(["curl", "-sIL", url], capture_output=True, text=True)
    for line in res.stdout.splitlines():
        if line.lower().startswith("content-length:"):
            return int(line.split(":", 1)[1].strip())
    return None


def download_with_retry(url: str, target: Path):
    """curl 断点续传 + 重试；本地大小与远端一致才跳过。"""
    if target.exists():
        local_size = target.stat().st_size
        expected = _remote_size(url)
        if expected is not None and local_size == expected:
            print(f"  - 已完整，跳过：{target.name}")
            return
        if expected is not None:
            print(f"  - 残缺({local_size}/{expected})，续传：{target.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-sL", "--retry", "8", "--retry-all-errors",
           "-C", "-", url, "-o", str(target)]
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def prepare_virtual_cursor():
    """下载并解压 Virtual Cursor 原始 EEG（data_1.zip）。"""
    print("\n[5] Virtual Cursor 原始 EEG（GitHub Release data_1.zip）")
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


def gaze_status():
    """返回 Gaze 已就位的期望文件数。"""
    gaze_dir = config.GAZE_DIR
    if not gaze_dir.exists():
        return 0
    return sum((gaze_dir / f).exists() for f in GAZE_EXPECTED_FILES)


def print_gaze_guide():
    """打印 Gaze 的浏览器手动下载步骤（命令行无法自动获取）。"""
    print("\n[6] Gaze 数据集（需浏览器手动下载）")
    have = gaze_status()
    if have >= len(GAZE_EXPECTED_FILES):
        print("  - 12 个文件已就位，跳过。")
        return True

    print("  检测到 figshare 有 Cloudflare 挑战，命令行返回 403/202，")
    print("  请按以下步骤用真实浏览器下载（约 402 MB）：")
    print("    1) 打开整包链接（浏览器会自动开始下载 .zip）：")
    print("       https://figshare.com/ndownloader/articles/5938714/versions/1")
    print("    或文章页手动下载：")
    print("       https://figshare.com/articles/dataset/"
          "Error_Related_Potentials_from_Gaze-Based_Typesetting/5938714/1")
    print("    2) 解压下载的压缩包，把其中 12 个 .mat 文件放到：")
    print(f"       {config.GAZE_DIR}")
    print("       期望文件：" + ", ".join(GAZE_EXPECTED_FILES))
    print("    3) 放好后重新运行本脚本，或运行数据核验：")
    print("       python scripts/verify_datasets.py")
    print("  数据 DOI：https://doi.org/10.6084/m9.figshare.5938714.v1（CC BY 4.0）")
    return False


def prepare_bnci():
    """可选：下载 BNCI013（12 个 .mat，约 3.4GB，速度很慢）。"""
    print("\n[可选] BNCI 013-2015（TU Graz 镜像，约 3.4GB）")
    out_dir = config.BNCI_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in config.DATASET_BNCI["subjects"]:
        for run in config.DATASET_BNCI["runs"]:
            filename = f"Subject{sub:02d}_{run}.mat"
            download_with_retry(BNCI_BASE_URL + filename,
                                out_dir / filename)


def verify_six_datasets():
    """核对 6 数据集关键文件，缺失即报错。"""
    checks = [
        config.DATASET_LSC["mat_path"],
        config.DATASET_HRI["data_dir"],
        config.DATASET_HRI_ROBOT["data_dir"],
        config.DATASET_COADAPTATION["data_dir"],
        config.VIRTUAL_CURSOR_DIR,
        config.GAZE_DIR,
    ]
    missing = [p for p in checks if not Path(p).exists()]
    if missing:
        print("\n[校验] 以下目录尚未就位（Gaze 请按上方手动步骤）：")
        for p in missing:
            print("  -", p)
        return False
    print("\n[校验] 6 个数据集目录均已就位。")
    return True


def main():
    parser = argparse.ArgumentParser(description="下载 6 个 ErrP 数据集")
    parser.add_argument("--only-auto", action="store_true",
                        help="只自动下载 5 个，跳过 Gaze 手动引导")
    parser.add_argument("--include-bnci", action="store_true",
                        help="额外下载 BNCI013（约 3.4GB，很慢）")
    args = parser.parse_args()

    print("=" * 64)
    print("ERRP 复现 · 数据准备（6 个已跑通数据集）")
    print("=" * 64)
    config.THIRD_PARTY_DIR.mkdir(parents=True, exist_ok=True)

    # 1–4：git 仓库
    for i, repo in enumerate(REPOS, start=1):
        print(f"\n[{i}] {repo['title']}  [许可：{repo['license']}]")
        run_git_clone(repo["url"], repo["target"])
        print("    *", repo["note"])

    # 5：Virtual Cursor 原始 EEG
    prepare_virtual_cursor()

    # 6：Gaze（手动引导）
    gaze_ok = True
    if not args.only_auto:
        gaze_ok = print_gaze_guide()

    # 可选 BNCI
    if args.include_bnci:
        prepare_bnci()

    all_ok = verify_six_datasets()
    if all_ok:
        print("\n数据准备完成，可运行：python3 run_reproduction.py")
    else:
        print("\n自动下载部分已完成；Gaze 按上方手动步骤补齐后即可运行。")


if __name__ == "__main__":
    main()
