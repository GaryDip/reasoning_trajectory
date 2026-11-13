#!/usr/bin/env python3
"""Read GPU memory and utilization without nvidia-smi, which hangs on this machine.

The driver and NVML are both fine here -- it is the `nvidia-smi` binary that hangs, even for
`nvidia-smi -i 0 --query-gpu=memory.used --format=csv`. So do NOT reset the GPUs or reload the
kernel module over it; that would kill other people's jobs for nothing. This script talks to
libnvidia-ml.so.1 directly through ctypes, which answers instantly.

The per-process list comes back empty because the other jobs live in different PID namespaces;
an empty list therefore means "cannot see them", never "nobody is using this card". Judge
occupancy by `util` and `used`, not by the process count.

    python cost/scripts/gpu_probe.py                 # one snapshot
    python cost/scripts/gpu_probe.py --watch 60      # sample every 60 s until Ctrl-C
    python cost/scripts/gpu_probe.py --watch 60 --csv cost/results/gpu_occupancy.csv
    python cost/scripts/gpu_probe.py --wait-idle 2 --max-util 5 --free-gb 100

`--wait-idle N` blocks until some card stays under `--max-util` with at least `--free-gb` free
for N consecutive samples, then prints its index and exits 0 (exit 1 on timeout) -- the point
being to grab an exclusive card for the cost benchmark the moment one frees up.
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import time
from datetime import datetime

GB = 1 << 30


class _Mem(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong)]


class _Util(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _Proc(ctypes.Structure):
    _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong),
                ("gpuInstanceId", ctypes.c_uint), ("computeInstanceId", ctypes.c_uint)]


class NVML:
    def __init__(self):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        if self.lib.nvmlInit_v2() != 0:
            sys.exit("nvmlInit failed")
        n = ctypes.c_uint()
        self.lib.nvmlDeviceGetCount_v2(ctypes.byref(n))
        self.n = n.value

    def snapshot(self) -> list[dict]:
        out = []
        for i in range(self.n):
            h = ctypes.c_void_p()
            self.lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h))
            m = _Mem()
            self.lib.nvmlDeviceGetMemoryInfo(h, ctypes.byref(m))
            u = _Util()
            ok = self.lib.nvmlDeviceGetUtilizationRates(h, ctypes.byref(u)) == 0
            cnt = ctypes.c_uint(0)
            self.lib.nvmlDeviceGetComputeRunningProcesses_v3(h, ctypes.byref(cnt), None)
            out.append({"gpu": i, "used_gb": m.used / GB, "free_gb": m.free / GB,
                        "total_gb": m.total / GB, "util": u.gpu if ok else None,
                        "n_visible_procs": cnt.value})
        return out


def fmt(rows: list[dict]) -> str:
    ts = datetime.now().strftime("%H:%M:%S")
    parts = [f"{ts}"]
    for r in rows:
        util = "  ?" if r["util"] is None else f"{r['util']:3d}"
        parts.append(f"GPU{r['gpu']} {r['used_gb']:5.1f}G {util}%")
    return "  |  ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=float, metavar="SEC", help="sample every SEC seconds")
    ap.add_argument("--csv", type=argparse.FileType("a"), help="append samples here")
    ap.add_argument("--wait-idle", type=int, metavar="N",
                    help="exit once a card has been idle for N consecutive samples")
    ap.add_argument("--max-util", type=int, default=5, help="idle means util <= this (default 5)")
    ap.add_argument("--free-gb", type=float, default=100.0,
                    help="idle also means at least this much free memory (default 100)")
    ap.add_argument("--timeout-min", type=float, default=0.0, help="0 = no timeout")
    args = ap.parse_args()

    nv = NVML()
    if args.csv:
        args.csv.write("ts,gpu,used_gb,free_gb,util\n")
    if not args.watch and not args.wait_idle:
        rows = nv.snapshot()
        w = max(len(f"{r['total_gb']:.0f}") for r in rows)
        print(f"{'GPU':>4} {'used':>9} {'free':>9} {'total':>{w+2}} {'util':>6} {'可见进程':>8}")
        for r in rows:
            u = "?" if r["util"] is None else f"{r['util']}%"
            print(f"{r['gpu']:>4} {r['used_gb']:8.1f}G {r['free_gb']:8.1f}G "
                  f"{r['total_gb']:{w}.0f}G {u:>6} {r['n_visible_procs']:>8}")
        idle = [r["gpu"] for r in rows
                if (r["util"] or 0) <= args.max_util and r["free_gb"] >= args.free_gb]
        print(f"\n按 util<={args.max_util}% 且空闲>={args.free_gb:.0f}G 判定空闲的卡："
              f"{idle if idle else '无'}")
        return 0

    interval = args.watch or 60.0
    streak: dict[int, int] = {}
    deadline = time.time() + args.timeout_min * 60 if args.timeout_min else None
    try:
        while True:
            rows = nv.snapshot()
            print(fmt(rows), flush=True)
            if args.csv:
                ts = datetime.now().isoformat(timespec="seconds")
                for r in rows:
                    args.csv.write(f"{ts},{r['gpu']},{r['used_gb']:.1f},"
                                   f"{r['free_gb']:.1f},{r['util']}\n")
                args.csv.flush()
            if args.wait_idle:
                for r in rows:
                    ok = (r["util"] or 0) <= args.max_util and r["free_gb"] >= args.free_gb
                    streak[r["gpu"]] = streak.get(r["gpu"], 0) + 1 if ok else 0
                    if streak[r["gpu"]] >= args.wait_idle:
                        print(f"\nGPU{r['gpu']} 连续 {args.wait_idle} 次采样空闲 "
                              f"(util<={args.max_util}%, 空闲 {r['free_gb']:.1f}G) -> 可以占用")
                        return 0
            if deadline and time.time() > deadline:
                print("\n超时，没有等到空闲卡")
                return 1
            time.sleep(interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
