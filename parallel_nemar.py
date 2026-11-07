# -*- coding: utf-8 -*-
"""
parallel_nemar.py
=================
并行下载 NEMAR ERP CORE（nm000132）flankers 任务的 EEGLAB 文件。

阶段 3 只需要与错误监控最相关的 flankers（ERN/LRP）任务：每个被试两个文件
（.set 元数据 + .fdt 二进制）。实测 data.nemar.org 单连接限速约 10–17 KB/s，
且并发过激进会返回 upstream timeout，因此默认 2 个 worker，配合断点续传、
期望大小校验与退避重试。

用法：
    python3 parallel_nemar.py                # 2 worker 下载 40 被试 flankers
    python3 parallel_nemar.py --workers 2
"""
import argparse
import subprocess
import threading
import time
from queue import Queue

import config

BASE_URL = "https://data.nemar.org/nm000132/v1.1.1"
OUT_DIR = config.NEMAR_DIR
SUBJECTS = [f"{i:03d}" for i in range(1, 41)]

# (被试, 后缀)；每个被试 .set + .fdt
FILES = [(s, ext) for s in SUBJECTS for ext in ("set", "fdt")]

print_lock = threading.Lock()
_size_cache = {}


def rel_path(sub, ext):
    return f"sub-{sub}/eeg/sub-{sub}_task-flankers_eeg.{ext}"


def remote_size(sub, ext):
    """
    HEAD 获取远端文件大小（字节），带缓存。

    NEMAR 限流时 HEAD 会返回 91 字节错误页而非真实大小，因此做合理性
    校验（.set 应 >100KB、.fdt 应 >10MB），不合规则退避后重试。
    """
    key = (sub, ext)
    if key in _size_cache:
        return _size_cache[key]
    min_plausible = 100_000 if ext == "set" else 10_000_000
    size = None
    for _ in range(6):
        out = subprocess.run(
            ["curl", "-sIL", f"{BASE_URL}/{rel_path(sub, ext)}"],
            capture_output=True, text=True)
        candidate = None
        for line in out.stdout.splitlines():
            if line.lower().startswith("content-length:"):
                candidate = int(line.split(":", 1)[1].strip())
        if candidate is not None and candidate >= min_plausible:
            size = candidate
            break
        time.sleep(5)
    _size_cache[key] = size
    return size


def download_one(sub, ext):
    """单文件断点续传 + 退避重试，直到本地大小等于期望大小。"""
    d = OUT_DIR / rel_path(sub, ext)
    d.parent.mkdir(parents=True, exist_ok=True)
    expected = remote_size(sub, ext)
    if expected is None:
        with print_lock:
            print(f"[!] sub-{sub} .{ext} 无法获取大小，跳过")
        return

    for attempt in range(1, 300):
        have = d.stat().st_size if d.exists() else 0
        if have == expected:
            with print_lock:
                print(f"[完成] sub-{sub} .{ext} {have/1e6:.1f}MB")
            return
        subprocess.run(
            ["curl", "-sL", "--retry", "5", "--retry-all-errors",
             "-C", "-", f"{BASE_URL}/{rel_path(sub, ext)}",
             "-o", str(d)])
        now = d.stat().st_size if d.exists() else 0
        # 无增长（限流/错误页）或极小错误页：退避，避免固化坏文件
        if now <= have or (now < 1000 and now != expected):
            if now < 1000:
                d.unlink(missing_ok=True)
            time.sleep(10)
        with print_lock:
            print(f"  [sub-{sub} .{ext}] 尝试{attempt} "
                  f"{now/1e6:.1f}/{expected/1e6:.1f}MB", flush=True)
    with print_lock:
        print(f"[!] sub-{sub} .{ext} 多次重试未完成")


def worker(q):
    while not q.empty():
        sub, ext = q.get()
        try:
            download_one(sub, ext)
        finally:
            q.task_done()


def main():
    parser = argparse.ArgumentParser(description="并行下载 NEMAR flankers")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    q = Queue()
    for item in FILES:
        q.put(item)

    threads = []
    for _ in range(args.workers):
        t = threading.Thread(target=worker, args=(q,), daemon=True)
        t.start()
        threads.append(t)

    while any(t.is_alive() for t in threads):
        time.sleep(30)
        have = sum(f.stat().st_size for f in OUT_DIR.glob("**/*.fdt"))
        have += sum(f.stat().st_size for f in OUT_DIR.glob("**/*.set"))
        print(f"### 总进度 {have/1e9:.2f} GB", flush=True)
    q.join()
    print("全部下载结束")


if __name__ == "__main__":
    main()
