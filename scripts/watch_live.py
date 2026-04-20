from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
POLL_SEC = 1.0
RECENT_SEC = 300
POWERSHELL_EXE = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
WATCH_KEYWORDS = ("watch_", "open_live_terminal", "watch_live.py")
LOG_PRIORITY = ("train.log", "sfm.log", "phase0.log", "launcher.out.log", "launcher.err.log")


def _clear_screen() -> None:
    os.system("cls")


def _run_root_for_log(path: Path) -> Path:
    return path.parent.parent


def _short_command(line: str, width: int = 120) -> str:
    text = " ".join(line.split())
    return text if len(text) <= width else text[: width - 3] + "..."


def _query_processes() -> list[dict[str, Any]]:
    if not POWERSHELL_EXE.exists():
        return []
    script = r"""
$items = Get-CimInstance Win32_Process |
  Where-Object {
    $_.Name -in @('python.exe','colmap.exe','glomap.exe','powershell.exe') -and
    $_.CommandLine -like '*3d-recon-pipeline*'
  } |
  Select-Object Name,ProcessId,ParentProcessId,CommandLine
$items | ConvertTo-Json -Compress
"""
    result = subprocess.run(
        [str(POWERSHELL_EXE), "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        return []
    payload = result.stdout.strip()
    if not payload:
        return []
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return []
    items = data if isinstance(data, list) else [data]
    filtered: list[dict[str, Any]] = []
    for item in items:
        cmd = str(item.get("CommandLine") or "")
        if any(keyword in cmd for keyword in WATCH_KEYWORDS):
            continue
        filtered.append(item)
    return filtered


def _candidate_logs() -> list[Path]:
    if not OUTPUTS_ROOT.exists():
        return []
    candidates: list[Path] = []
    for path in OUTPUTS_ROOT.rglob("*.log"):
        if path.name in LOG_PRIORITY:
            candidates.append(path)
    return candidates


def _latest_active_log() -> Path | None:
    processes = _query_processes()
    if not processes:
        return None
    now = time.time()
    recent_logs = [
        path
        for path in _candidate_logs()
        if path.exists() and (now - path.stat().st_mtime) <= RECENT_SEC
    ]
    if not recent_logs:
        return None
    recent_logs.sort(
        key=lambda p: (
            p.name not in LOG_PRIORITY,
            LOG_PRIORITY.index(p.name) if p.name in LOG_PRIORITY else 999,
            -p.stat().st_mtime,
        )
    )
    # Prefer the most recently updated log among prioritized names.
    return max(
        recent_logs,
        key=lambda p: (
            p.stat().st_mtime,
            -LOG_PRIORITY.index(p.name) if p.name in LOG_PRIORITY else -999,
        ),
    )


def _print_dashboard(processes: list[dict[str, Any]], current_log: Path | None) -> None:
    _clear_screen()
    print("3d-recon-pipeline live monitor")
    print("")
    print(f"Current log : {current_log if current_log else '-'}")
    print(f"Run        : {_run_root_for_log(current_log).name if current_log else '-'}")
    print(f"Status     : {'running' if processes else 'idle'}")
    print("")
    print("Processes")
    print("-" * 72)
    if not processes:
        print("No active pipeline process.")
    else:
        for item in processes:
            print(f"{item['Name']:<14} pid={item['ProcessId']:<8} ppid={item['ParentProcessId']:<8}")
            print(f"  {_short_command(str(item.get('CommandLine') or ''))}")
    print("")
    print("Watching rule")
    print("-" * 72)
    print("If a recent log exists, switch to tail mode automatically.")
    print("Otherwise keep this process panel.")


def _tail_file(path: Path) -> None:
    _clear_screen()
    print(f"Watching log : {path}")
    print(f"Run          : {_run_root_for_log(path).name}")
    print("")

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        try:
            lines = handle.readlines()
        except Exception:
            lines = []
        for line in lines[-40:]:
            print(line.rstrip())

        while True:
            line = handle.readline()
            if line:
                print(line.rstrip(), flush=True)
                continue

            next_log = _latest_active_log()
            if next_log != path:
                return
            time.sleep(POLL_SEC)


def main() -> None:
    while True:
        processes = _query_processes()
        current_log = _latest_active_log()
        if current_log is not None and current_log.exists():
            _tail_file(current_log)
            continue
        _print_dashboard(processes, current_log)
        time.sleep(2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
