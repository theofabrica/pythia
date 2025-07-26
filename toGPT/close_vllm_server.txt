#!/usr/bin/env python3
import os
import signal
import subprocess
import psutil


def find_vllm_server_process():
    for proc in psutil.process_iter(['pid', 'cmdline']):
        if proc.info['cmdline'] and 'vllm_chat_server.py' in ' '.join(proc.info['cmdline']):
            return proc.pid
    return None


def main():
    pid = find_vllm_server_process()
    if pid:
        print(f"Arrêt du serveur vLLM (PID {pid})...")
        os.kill(pid, signal.SIGTERM)
    else:
        print("Aucun serveur vLLM trouvé.")

    print("Flush des GPUs...")
    subprocess.run(["flush_gpus"])


if __name__ == "__main__":
    main()
