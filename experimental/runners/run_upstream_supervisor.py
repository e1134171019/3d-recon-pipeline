from __future__ import annotations

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


app = typer.Typer(help="監控 upstream suite，若異常退出則自動重啟。")
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATRIX = PROJECT_ROOT / "outputs" / "experiments" / "upstream_2x2_matrix.json"


def _read_json(path: Path) -> dict[str, Any]:
    encodings = ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "cp950", "mbcs")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return json.loads(path.read_text(encoding=encoding))
        except Exception as exc:
            last_error = exc
    raise last_error if last_error is not None else ValueError(f"無法解析 JSON：{path}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _matrix_root(matrix: Path) -> Path:
    payload = _read_json(matrix)
    shared = payload.get("shared", {})
    outdir_root = shared.get("outdir_root")
    if outdir_root:
        root = Path(str(outdir_root))
        return root if root.is_absolute() else (PROJECT_ROOT / root)
    return matrix.with_suffix("")


def _resolve_python() -> Path:
    preferred = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    return preferred if preferred.exists() else Path(sys.executable)


def _status_path(matrix: Path) -> Path:
    return _matrix_root(matrix) / "suite_logs" / "upstream_suite_status.json"


def _supervisor_status_path(matrix: Path) -> Path:
    return _matrix_root(matrix) / "suite_logs" / "upstream_supervisor_status.json"


def _launch_once(matrix: Path) -> subprocess.Popen[str]:
    suite_logs = _matrix_root(matrix) / "suite_logs"
    suite_logs.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_log = suite_logs / f"upstream_supervised_{ts}.out.log"
    err_log = suite_logs / f"upstream_supervised_{ts}.err.log"
    stdout = out_log.open("w", encoding="utf-8", errors="replace")
    stderr = err_log.open("w", encoding="utf-8", errors="replace")
    process = subprocess.Popen(
        [str(_resolve_python()), "-m", "src.run_upstream_suite", "--matrix", str(matrix.resolve())],
        cwd=str(PROJECT_ROOT),
        stdout=stdout,
        stderr=stderr,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return process


def _current_status(matrix: Path) -> dict[str, Any]:
    path = _status_path(matrix)
    if not path.exists():
        return {"status": "missing", "current": None, "completed": [], "failed": []}
    return _read_json(path)


def _write_supervisor_status(matrix: Path, payload: dict[str, Any]) -> None:
    _write_json(_supervisor_status_path(matrix), payload)


@app.command()
def run(
    matrix: Path = typer.Option(DEFAULT_MATRIX, help="上游實驗矩陣 JSON"),
    max_restarts: int = typer.Option(5, help="異常退出時最多自動重啟次數"),
    restart_delay_sec: int = typer.Option(5, help="重啟前等待秒數"),
) -> None:
    matrix_path = matrix if matrix.is_absolute() else (PROJECT_ROOT / matrix)
    if not matrix_path.exists():
        raise typer.BadParameter(f"找不到矩陣：{matrix_path}")

    restarts = 0

    try:
        while True:
            process = _launch_once(matrix_path)
            launch_time = datetime.now().isoformat(timespec="seconds")
            _write_supervisor_status(
                matrix_path,
                {
                    "status": "running",
                    "timestamp": launch_time,
                    "restarts": restarts,
                    "pid": process.pid,
                    "current_suite_pid": process.pid,
                    "last_launch_time": launch_time,
                },
            )
            console.print(
                Panel(
                    f"啟動 upstream suite\npid={process.pid}\nrestarts={restarts}",
                    title="[bold cyan]Supervisor[/]",
                    border_style="cyan",
                )
            )

            return_code = process.wait()
            status = _current_status(matrix_path)
            suite_status = str(status.get("status", "unknown"))
            current = status.get("current")

            if suite_status in {"completed", "completed_with_failures"}:
                _write_supervisor_status(
                    matrix_path,
                    {
                        "status": "finished",
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "restarts": restarts,
                        "return_code": return_code,
                        "last_exit_code": return_code,
                        "suite_status": suite_status,
                        "current": current,
                    },
                )
                break

            if restarts >= max_restarts:
                _write_supervisor_status(
                    matrix_path,
                    {
                        "status": "gave_up",
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "restarts": restarts,
                        "return_code": return_code,
                        "last_exit_code": return_code,
                        "suite_status": suite_status,
                        "current": current,
                    },
                )
                break

            restarts += 1
            _write_supervisor_status(
                matrix_path,
                {
                    "status": "restarting",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "restarts": restarts,
                    "return_code": return_code,
                    "last_exit_code": return_code,
                    "suite_status": suite_status,
                    "current": current,
                },
            )
            time.sleep(restart_delay_sec)
    except Exception as exc:
        _write_supervisor_status(
            matrix_path,
            {
                "status": "crashed",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "restarts": restarts,
                "error": str(exc),
            },
        )
        raise


if __name__ == "__main__":
    app()
