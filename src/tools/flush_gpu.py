#!/usr/bin/env python3
"""
flush_gpu.py – Libère proprement la mémoire GPU et tue les processus Python/CUDA
qui tournent **dans le conteneur**.

• Torch : vidage du cache + collecte IPC.
• NVIDIA-SMI : repère les PIDs qui consomment du GPU et les termine,
  en s’assurant qu’ils appartiennent bien au conteneur courant.
"""

import gc
import os
import subprocess
import sys
import time

import psutil
import torch


def _flush_torch():
    if not torch.cuda.is_available():
        return
    print("🧽  Releasing PyTorch memory…")
    for idx in range(torch.cuda.device_count()):
        print(f" - device {idx}")
        torch.cuda.set_device(idx)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def _kill_python_gpu_procs():
    print("💀  Killing Python GPU processes…")
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            encoding="utf-8",
        )
        pids = {int(p.strip()) for p in output.splitlines() if p.strip()}
    except Exception as err:
        print(f"⚠️  nvidia-smi unavailable: {err}")
        return

    me = os.getpid()
    for pid in pids:
        if pid == me:
            continue
        try:
            proc = psutil.Process(pid)
            # sécurité : ne tue que les cmdlines localisées dans /app (conteneur)
            if any("/app" in arg for arg in proc.cmdline()):
                proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    time.sleep(1)


def main():
    _flush_torch()
    _kill_python_gpu_procs()
    print("✅  GPU memory flush complete.")


if __name__ == "__main__":
    main()
