import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(r"C:\3d-recon-pipeline\outputs\experiments\upstream_2x2_matrix")
STATUS_PATH = ROOT / "suite_logs" / "upstream_suite_status.json"
POLL_SEC = 1.0


def read_json(path: Path):
    if not path.exists():
        return None
    for encoding in ("utf-8", "utf-8-sig", "cp950"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except Exception:
            continue
    return None


def count_files(path: Path) -> int | None:
    if not path.exists():
        return None
    return sum(1 for item in path.iterdir() if item.is_file())


def latest_log(experiment_root: Path) -> Path | None:
    log_dir = experiment_root / "logs"
    if not log_dir.exists():
        return None
    candidates = [log_dir / "train.log", log_dir / "sfm.log", log_dir / "phase0.log"]
    candidates = [path for path in candidates if path.exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def print_header(current: str, stage: str, status: str):
    print("=" * 72, flush=True)
    print(f"Current : {current or '-'}", flush=True)
    print(f"Stage   : {stage or '-'}", flush=True)
    print(f"Status  : {status or '-'}", flush=True)
    print("=" * 72, flush=True)


def stream_log(path: Path, current: str, stage: str, status: str):
    print_header(current, stage, status)
    print(f"Watching log: {path}", flush=True)
    print("", flush=True)

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(0, os.SEEK_END)
        while True:
            line = handle.readline()
            if line:
                print(line.rstrip(), flush=True)
                continue

            payload = read_json(STATUS_PATH) or {}
            next_current = str(payload.get("current") or "")
            next_stage = str(payload.get("stage") or "")
            next_status = str(payload.get("status") or "")
            next_experiment_root = ROOT / next_current if next_current else None
            next_log = latest_log(next_experiment_root) if next_experiment_root else None

            if (
                next_current != current
                or next_stage != stage
                or next_status != status
                or next_log != path
            ):
                print("", flush=True)
                print("[switch]", flush=True)
                return

            time.sleep(POLL_SEC)


def monitor_phase0(experiment_root: Path, current: str, stage: str, status: str):
    phase0_root = experiment_root / "phase0"
    candidates_dir = phase0_root / "candidates"
    clean_dir = phase0_root / "frames_cleaned"
    small_dir = phase0_root / "frames_1600"

    last_snapshot = None
    print_header(current, stage, status)
    print("Watching phase0 progress", flush=True)
    print("", flush=True)

    while True:
        payload = read_json(STATUS_PATH) or {}
        next_current = str(payload.get("current") or "")
        next_stage = str(payload.get("stage") or "")
        next_status = str(payload.get("status") or "")
        if next_current != current or next_stage != stage or next_status != status:
            print("", flush=True)
            print("[switch]", flush=True)
            return

        snapshot = (
            count_files(candidates_dir),
            count_files(clean_dir),
            count_files(small_dir),
            sum(1 for _ in [*os.popen("tasklist /FI \"IMAGENAME eq python.exe\" /FO CSV /NH"), *os.popen("tasklist /FI \"IMAGENAME eq colmap.exe\" /FO CSV /NH")]),
        )
        if snapshot != last_snapshot:
            candidates_count, clean_count, small_count, proc_rows = snapshot
            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"candidates={candidates_count if candidates_count is not None else 'missing'} "
                f"frames_cleaned={clean_count if clean_count is not None else 'missing'} "
                f"frames_1600={small_count if small_count is not None else 'missing'} "
                f"process_rows={proc_rows}",
                flush=True,
            )
            last_snapshot = snapshot
        time.sleep(POLL_SEC)


def main():
    print("Upstream live monitor ready.", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    print("", flush=True)

    while True:
        payload = read_json(STATUS_PATH) or {}
        current = str(payload.get("current") or "")
        stage = str(payload.get("stage") or "")
        status = str(payload.get("status") or "")

        if not current:
            print_header("-", "-", status or "idle")
            print("No active experiment.", flush=True)
            time.sleep(3)
            continue

        experiment_root = ROOT / current
        log_path = latest_log(experiment_root)

        if stage == "phase0" or log_path is None:
            monitor_phase0(experiment_root, current, stage, status)
        else:
            stream_log(log_path, current, stage, status)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
