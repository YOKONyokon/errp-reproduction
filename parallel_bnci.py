# -*- coding: utf-8 -*-
"""
parallel_bnci.py
================
并行下载 BNCI 013-2015（TU Graz lampx 镜像）。

背景：实测该服务器对单条 TCP 连接限速（约 10–15 KB/s），但允许少量
并发连接，2 个 worker 并行可把总吞吐提升约 3–4 倍。本脚本用线程池
管理 2 个下载 worker，每个文件独立做断点续传、期望大小校验与重试。

用法：
    python3 parallel_bnci.py              # 2 worker 并行下载全部 12 文件
    python3 parallel_bnci.py --workers 3  # 指定并发数
"""

import argparse
import subprocess
import threading
import time
from pathlib import Path
from queue import Queue

import config

BASE_URL = "https://lampx.tugraz.at/~bci/database/013-2015"
OUT_DIR = config.BNCI_DIR

# 12 个文件：6 被试 × 2 session
FILES = [f"Subject0{s}_{r}" for s in range(1, 7) for r in ("s1", "s2")]

# 已实测的完整文件大小（用于完整性校验；缺失时退化为 HEAD 动态获取）
KNOWN_SIZE = {
    "Subject01_s1": 269644558,
    "Subject01_s2": 261040751,
    "Subject02_s1": 265085116,
}

print_lock = threading.Lock()


def remote_size(name):
    """用 HEAD 请求获取远端文件大小（字节）。"""
    if name in KNOWN_SIZE:
        return KNOWN_SIZE[name]
    out = subprocess.run(
        ["curl", "-sIL", f"{BASE_URL}/{name}.mat"],
        capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.lower().startswith("content-length:"):
            return int(line.split(":", 1)[1].strip())
    return None


def download_one(name):
    """下载单个文件，断点续传 + 重试，直到本地大小等于期望大小。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUT_DIR / f"{name}.mat"
    expected = remote_size(name)
    if expected is None:
        with print_lock:
            print(f"[!] {name} 无法获取远端大小，跳过")
        return

    for attempt in range(1, 200):
        have = target.stat().st_size if target.exists() else 0
        if have == expected:
            with print_lock:
                print(f"[完成] {name} {have/1e6:.1f}MB")
            return
        # 用 curl -C - 从当前位置续传；--retry 处理瞬时错误
        subprocess.run(
            ["curl", "-sL", "--retry", "5", "--retry-all-errors",
             "-C", "-", f"{BASE_URL}/{name}.mat", "-o", str(target)])
        with print_lock:
            now = target.stat().st_size if target.exists() else 0
            print(f"  [{name}] 尝试{attempt} {now/1e6:.1f}/{expected/1e6:.1f}MB")
    with print_lock:
        print(f"[!] {name} 多次重试仍未完成")


def worker(q):
    """线程 worker：不断从队列取文件并下载。"""
    while not q.empty():
        name = q.get()
        try:
            download_one(name)
        finally:
            q.task_done()


def main():
    parser = argparse.ArgumentParser(description="并行下载 BNCI 013")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    q = Queue()
    for name in FILES:
        q.put(name)

    threads = []
    for _ in range(args.workers):
        t = threading.Thread(target=worker, args=(q,), daemon=True)
        t.start()
        threads.append(t)

    # 主线程定期打印总进度
    total_expected = sum(remote_size(n) or 0 for n in FILES)
    while any(t.is_alive() for t in threads):
        time.sleep(20)
        have = sum(f.stat().st_size for f in OUT_DIR.glob("Subject*.mat"))
        print(f"### 总进度 {have/1e9:.2f}/{total_expected/1e9:.2f}GB "
              f"({100*have/total_expected:.1f}%)", flush=True)
    q.join()
    print("全部下载结束")


if __name__ == "__main__":
    main()
