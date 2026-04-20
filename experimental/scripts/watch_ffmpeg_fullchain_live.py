from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


RUNS_ROOT = Path(r"C:\3d-recon-pipeline\outputs\runs")
POLL_SEC = 1.0
POWERSHELL_EXE = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"


def latest_run_root() -> Path | None:
    candidates = [p for p in RUNS_ROOT.iterdir() if p.is_dir() and p.name.startswith("ffmpeg_fullchain_")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def latest_stage_log(run_root: Path) -> Path | None:
    log_dir = run_root / "logs"
    if not log_dir.exists():
        return None
    preferred = [log_dir / "train.log", log_dir / "sfm.log", log_dir / "phase0.log"]
    for path in preferred:
        if path.exists() and path.stat().st_size > 0:
            return path
    fallback = log_dir / "launcher.out.log"
    return fallback if fallback.exists() else None


def process_alive() -> bool:
    if not POWERSHELL_EXE.exists():
        return False
    script = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*run_ffmpeg_fullchain*' } | "
        "ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        [str(POWERSHELL_EXE), "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        return False
    payload = result.stdout.strip()
    if not payload:
        return False
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return False
    if isinstance(data, list):
        return len(data) > 0
    return isinstance(data, dict) and bool(data)


def print_header(run_root: Path | None, log_path: Path | None):
    print("=" * 72, flush=True)
    print(f"Run    : {run_root.name if run_root else '-'}", flush=True)
    print(f"Status : {'running' if process_alive() else 'idle'}", flush=True)
    print(f"Log    : {log_path.name if log_path else '-'}", flush=True)
    print("=" * 72, flush=True)


def stream_log(run_root: Path, log_path: Path):
    print_header(run_root, log_path)
    print(f"Watching log: {log_path}", flush=True)
    print("", flush=True)

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(0, os.SEEK_END)
        while True:
            line = handle.readline()
            if line:
                print(line.rstrip(), flush=True)
                continue

            next_root = latest_run_root()
            next_log = latest_stage_log(next_root) if next_root else None
            if next_root != run_root or next_log != log_path:
                print("", flush=True)
                print("[switch]", flush=True)
                return
            time.sleep(POLL_SEC)


def main():
    print("ffmpeg full-chain live monitor ready.", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    print("", flush=True)

    while True:
        run_root = latest_run_root()
        if run_root is None:
            print_header(None, None)
            print("No ffmpeg full-chain run found.", flush=True)
            time.sleep(3)
            continue

        log_path = latest_stage_log(run_root)
        if log_path is None:
            print_header(run_root, None)
            print("No log yet.", flush=True)
            time.sleep(2)
            continue

        stream_log(run_root, log_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
