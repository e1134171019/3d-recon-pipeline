from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import typer

app = typer.Typer(help="Sequential suite runner for Mask Route A variants.")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "mask_route_a"
VALID_VARIANTS = ("A1_highlight_mask", "A2_machine_roi", "A3_combined")


def _resolve_python() -> Path:
    preferred = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    return preferred if preferred.exists() else Path(sys.executable)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _wait_for_run_completion(run_root: Path, poll_seconds: int) -> None:
    summary = run_root / "run_summary.json"
    route_log = run_root / "logs" / "route.log"

    print(f"[WAIT] Waiting for run to finish: {run_root}", flush=True)
    while not summary.exists():
        if route_log.exists():
            try:
                tail = route_log.read_text(encoding="utf-8", errors="replace").splitlines()[-1]
                print(f"[WAIT] last log: {tail}", flush=True)
            except Exception:
                pass
        time.sleep(poll_seconds)
    print(f"[OK] Finished run detected: {summary}", flush=True)


def _run_variant(variant: str, run_root: Path) -> None:
    python = str(_resolve_python())
    cmd = [python, "-u", "-m", "src.run_mask_route_opencv", "--variant", variant, "--run-root", str(run_root)]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    print(f"[RUN] {variant}", flush=True)
    print(">> " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=str(PROJECT_ROOT), env=env)
    if rc != 0:
        raise SystemExit(rc)


@app.command()
def main(
    variants: list[str] = typer.Argument(..., help="Variants to run sequentially."),
    wait_for_run_root: str = typer.Option("", help="Wait for an active run to finish before starting."),
    poll_seconds: int = typer.Option(20, help="Polling interval while waiting."),
) -> None:
    for variant in variants:
        if variant not in VALID_VARIANTS:
            raise SystemExit(f"Unsupported variant: {variant}. Expected one of: {VALID_VARIANTS}")

    if wait_for_run_root:
        wait_root = Path(wait_for_run_root)
        if not wait_root.is_absolute():
            wait_root = (PROJECT_ROOT / wait_root).resolve()
        if not wait_root.exists():
            raise SystemExit(f"wait_for_run_root not found: {wait_root}")
        _wait_for_run_completion(wait_root, poll_seconds)

    for variant in variants:
        run_root = DEFAULT_ROOT / f"{variant}_{_timestamp()}"
        _run_variant(variant, run_root)

    print("[OK] Mask Route suite finished.", flush=True)


if __name__ == "__main__":
    app()
