import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel

from src.run_experiment_matrix import (
    _find_experiment,
    _load_matrix,
    _materialize_all,
    _matrix_name,
    _matrix_root,
)


app = typer.Typer(help="依序執行整組 A 路線實驗")
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATRIX = PROJECT_ROOT / "outputs" / "experiments" / "a_route_matrix.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _final_checkpoint_path(experiment_root: Path, iterations: int) -> Path:
    return experiment_root / "3DGS_models" / "ckpts" / f"ckpt_{iterations - 1}_rank0.pt"


def _train_log_path(experiment_root: Path) -> Path:
    return experiment_root / "logs" / "train.log"


def _is_log_recent(path: Path, recent_seconds: int) -> bool:
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age <= recent_seconds


def _active_experiments(
    payload: dict[str, Any],
    root: Path,
    recent_seconds: int,
) -> list[str]:
    active: list[str] = []
    for exp in payload.get("experiments", []):
        name = str(exp["name"])
        iterations = int(exp.get("params", {}).get("iterations", 30000))
        experiment_root = root / name
        if _final_checkpoint_path(experiment_root, iterations).exists():
            continue
        if _is_log_recent(_train_log_path(experiment_root), recent_seconds):
            active.append(name)
    return active


def _run_matrix_experiment(name: str, matrix: Path) -> int:
    python_exe = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if not python_exe.exists():
        python_exe = Path(sys.executable)
    cmd = [
        str(python_exe),
        "-m",
        "src.run_experiment_matrix",
        "run",
        "--matrix",
        str(matrix),
        "--name",
        name,
    ]
    console.print(f"[cyan]啟動實驗[/] {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode


def _collect(matrix: Path) -> None:
    python_exe = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if not python_exe.exists():
        python_exe = Path(sys.executable)
    subprocess.run(
        [str(python_exe), "-m", "src.run_experiment_matrix", "collect", "--matrix", str(matrix)],
        cwd=PROJECT_ROOT,
        check=False,
    )


def _wait_until_done(
    name: str,
    experiment_root: Path,
    iterations: int,
    recent_seconds: int,
    idle_timeout_seconds: int,
    suite_status_path: Path,
) -> bool:
    ckpt = _final_checkpoint_path(experiment_root, iterations)
    train_log = _train_log_path(experiment_root)
    last_recent_ts = time.time() if _is_log_recent(train_log, recent_seconds) else 0.0

    while True:
        if ckpt.exists():
            console.print(f"[green]已完成[/] {name} -> {ckpt.name}")
            return True

        if _is_log_recent(train_log, recent_seconds):
            last_recent_ts = time.time()
            console.print(f"[yellow]等待現有執行中的實驗完成[/] {name}")
            time.sleep(30)
            continue

        if last_recent_ts and (time.time() - last_recent_ts) <= idle_timeout_seconds:
            console.print(f"[yellow]Log 暫時未更新，持續等待[/] {name}")
            time.sleep(30)
            continue

        _write_json(
            suite_status_path,
            {
                "status": "stalled",
                "experiment": name,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "train_log": str(train_log.resolve()),
            },
        )
        console.print(f"[red]偵測到實驗停滯[/] {name}")
        return False


@app.command()
def run(
    matrix: Path = typer.Option(DEFAULT_MATRIX, help="實驗矩陣 JSON"),
    recent_seconds: int = typer.Option(120, help="log 視為仍在執行中的秒數"),
    idle_timeout_seconds: int = typer.Option(900, help="超過多久無更新視為停滯"),
) -> None:
    matrix_path, payload = _load_matrix(matrix)
    _materialize_all(matrix_path, payload)
    root = _matrix_root(matrix_path, payload)
    suite_log_root = root / "suite_logs"
    suite_log_root.mkdir(parents=True, exist_ok=True)
    suite_status_path = suite_log_root / "suite_status.json"

    sequence = [str(exp["name"]) for exp in payload.get("experiments", [])]
    _write_json(
        suite_status_path,
        {
            "status": "running",
            "matrix_name": _matrix_name(matrix_path, payload),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "sequence": sequence,
            "current": None,
            "completed": [],
            "failed": [],
        },
    )

    completed: list[str] = []
    failed: list[str] = []

    for name in sequence:
        experiment = _find_experiment(payload, name)
        params = experiment.get("params", {})
        iterations = int(params.get("iterations", 30000))
        experiment_root = root / name
        ckpt = _final_checkpoint_path(experiment_root, iterations)
        train_log = _train_log_path(experiment_root)

        status_payload = {
            "status": "running",
            "matrix_name": _matrix_name(matrix_path, payload),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "current": name,
            "completed": completed,
            "failed": failed,
        }
        _write_json(suite_status_path, status_payload)

        if ckpt.exists():
            console.print(f"[green]略過已完成組別[/] {name}")
            completed.append(name)
            _collect(matrix_path)
            continue

        while True:
            active_now = _active_experiments(payload, root, recent_seconds)
            if not active_now:
                break
            if active_now == [name]:
                break
            console.print(f"[yellow]偵測到其他實驗仍在執行，先等待[/] {', '.join(active_now)}")
            _write_json(
                suite_status_path,
                {
                    "status": "waiting_other_experiment",
                    "matrix_name": _matrix_name(matrix_path, payload),
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "current": name,
                    "active_now": active_now,
                    "completed": completed,
                    "failed": failed,
                },
            )
            time.sleep(30)

        if _is_log_recent(train_log, recent_seconds):
            ok = _wait_until_done(
                name,
                experiment_root,
                iterations,
                recent_seconds,
                idle_timeout_seconds,
                suite_status_path,
            )
            if ok:
                completed.append(name)
                _collect(matrix_path)
                continue
            failed.append(name)
            continue

        exit_code = _run_matrix_experiment(name, matrix_path)
        if exit_code != 0:
            console.print(f"[red]實驗失敗[/] {name} exit={exit_code}")
            failed.append(name)
            _collect(matrix_path)
            continue

        completed.append(name)
        _collect(matrix_path)

    final_status = "completed" if not failed else "completed_with_failures"
    _write_json(
        suite_status_path,
        {
            "status": final_status,
            "matrix_name": _matrix_name(matrix_path, payload),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "completed": completed,
            "failed": failed,
        },
    )

    console.print(
        Panel(
            f"完成組別: {', '.join(completed) if completed else '無'}\n"
            f"失敗組別: {', '.join(failed) if failed else '無'}\n"
            f"狀態檔: {suite_status_path.resolve()}",
            title="[bold green]A 路線實驗序列完成[/]",
            border_style="green" if not failed else "yellow",
        )
    )


if __name__ == "__main__":
    app()
