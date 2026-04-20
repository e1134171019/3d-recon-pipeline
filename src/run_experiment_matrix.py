import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


app = typer.Typer(help="A 路線參數實驗矩陣 runner")
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATRIX = PROJECT_ROOT / "outputs" / "experiments" / "a_route_matrix.json"
TRAIN_KEYS = {
    "imgdir",
    "colmap",
    "outdir",
    "iterations",
    "sh_degree",
    "densify_until",
    "scene_scale",
    "scale_json",
    "eval_steps",
    "data_factor",
    "absgrad",
    "grow_grad2d",
    "antialiased",
    "random_bkgd",
    "validation_report",
}


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


def _resolve_path(value: str | None) -> str:
    if not value:
        return ""
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)
    return str((PROJECT_ROOT / candidate).resolve())


def _load_matrix(matrix: Path) -> tuple[Path, dict[str, Any]]:
    matrix_path = matrix if matrix.is_absolute() else (PROJECT_ROOT / matrix)
    if not matrix_path.exists():
        raise typer.BadParameter(f"找不到矩陣檔：{matrix_path}")

    payload = _read_json(matrix_path)
    experiments = payload.get("experiments", [])
    if not isinstance(experiments, list) or not experiments:
        raise typer.BadParameter(f"矩陣格式不正確：{matrix_path}")
    return matrix_path, payload


def _matrix_name(matrix_path: Path, payload: dict[str, Any]) -> str:
    return str(payload.get("matrix_name") or matrix_path.stem)


def _matrix_root(matrix_path: Path, payload: dict[str, Any]) -> Path:
    shared = payload.get("shared", {})
    outdir_root = shared.get("outdir_root")
    if outdir_root:
        root = Path(str(outdir_root))
        return root if root.is_absolute() else (PROJECT_ROOT / root)
    return matrix_path.with_suffix("")


def _find_experiment(payload: dict[str, Any], name: str) -> dict[str, Any]:
    for experiment in payload.get("experiments", []):
        if experiment.get("name") == name:
            return experiment
    names = ", ".join(exp.get("name", "?") for exp in payload.get("experiments", []))
    raise typer.BadParameter(f"找不到實驗組 {name}。可用組別：{names}")


def _build_plan(
    matrix_path: Path,
    payload: dict[str, Any],
    experiment: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    shared = dict(payload.get("shared", {}))
    params = dict(shared)
    params.update(experiment.get("params", {}))

    experiment_root = _matrix_root(matrix_path, payload) / str(experiment["name"])
    params["outdir"] = str((experiment_root / "3DGS_models").resolve())

    for key in ("imgdir", "colmap", "validation_report", "scale_json"):
        params[key] = _resolve_path(params.get(key))

    recommended_params = {key: params[key] for key in TRAIN_KEYS if key in params}
    plan = {
        "train_params": {
            "profile_name": experiment["name"],
            "matrix_name": _matrix_name(matrix_path, payload),
            "question": experiment.get("question", ""),
            "comparison_role": experiment.get("comparison_role", ""),
            "hypothesis": experiment.get("hypothesis", ""),
            "notes": experiment.get("notes", ""),
            "recommended_params": recommended_params,
        }
    }
    return plan, experiment_root


def _materialize_all(matrix_path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    root = _matrix_root(matrix_path, payload)
    root.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, Any]] = []

    for experiment in payload.get("experiments", []):
        plan, experiment_root = _build_plan(matrix_path, payload, experiment)
        params_path = experiment_root / "train_params.json"
        observation_path = experiment_root / "observation_template.json"
        meta_path = experiment_root / "experiment_meta.json"

        _write_json(params_path, plan)
        _write_json(
            observation_path,
            {
                "experiment_name": experiment["name"],
                "unity_subjective": "",
                "prune_needed": None,
                "notes": "",
            },
        )
        _write_json(
            meta_path,
            {
                "matrix_name": _matrix_name(matrix_path, payload),
                "experiment_name": experiment["name"],
                "question": experiment.get("question", ""),
                "comparison_role": experiment.get("comparison_role", ""),
                "hypothesis": experiment.get("hypothesis", ""),
                "recommended_params": plan["train_params"]["recommended_params"],
            },
        )
        generated.append(
            {
                "name": experiment["name"],
                "root": str(experiment_root.resolve()),
                "params_json": str(params_path.resolve()),
            }
        )

    _write_json(
        root / "manifest.json",
        {
            "matrix_name": _matrix_name(matrix_path, payload),
            "matrix_path": str(matrix_path.resolve()),
            "materialized_at": datetime.now().isoformat(timespec="seconds"),
            "experiments": generated,
        },
    )
    return generated


def _resolve_python() -> Path:
    preferred = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    return preferred if preferred.exists() else Path(sys.executable)


def _latest_val_stats(stats_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    if not stats_dir.exists():
        return None
    for path in stats_dir.glob("val_step*.json"):
        match = re.search(r"val_step(\d+)\.json$", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _stream_subprocess(cmd: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_encoding = sys.stdout.encoding or "utf-8"
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            safe_line = line.rstrip().encode(stdout_encoding, errors="replace").decode(
                stdout_encoding, errors="replace"
            )
            try:
                print(safe_line)
            except UnicodeEncodeError:
                print(safe_line.encode(stdout_encoding, errors="replace").decode(stdout_encoding, errors="replace"))
        return process.wait()


def _load_observation(experiment_root: Path) -> dict[str, Any]:
    observation_path = experiment_root / "observation_template.json"
    if observation_path.exists():
        return _read_json(observation_path)
    return {"unity_subjective": "", "prune_needed": None, "notes": ""}


@app.command(name="list")
def list_matrix(
    matrix: Path = typer.Option(DEFAULT_MATRIX, help="實驗矩陣 JSON"),
) -> None:
    matrix_path, payload = _load_matrix(matrix)
    table = Table(title=f"A 路線實驗矩陣：{_matrix_name(matrix_path, payload)}")
    table.add_column("Name", style="cyan")
    table.add_column("Role", style="green")
    table.add_column("Question", style="white")
    table.add_column("Params", style="magenta")

    for experiment in payload.get("experiments", []):
        params = experiment.get("params", {})
        summary = (
            f"iter={params.get('iterations')} | "
            f"densify={params.get('densify_until')} | "
            f"absgrad={params.get('absgrad', False)} | "
            f"grow={params.get('grow_grad2d', '-') } | "
            f"aa={params.get('antialiased', False)}"
        )
        table.add_row(
            str(experiment.get("name", "")),
            str(experiment.get("comparison_role", "")),
            str(experiment.get("question", "")),
            summary,
        )

    console.print(table)


@app.command()
def materialize(
    matrix: Path = typer.Option(DEFAULT_MATRIX, help="實驗矩陣 JSON"),
) -> None:
    matrix_path, payload = _load_matrix(matrix)
    generated = _materialize_all(matrix_path, payload)
    console.print(
        Panel(
            "\n".join(
                f"{item['name']}: {item['params_json']}"
                for item in generated
            ),
            title="[bold green]已生成實驗參數檔[/]",
            border_style="green",
        )
    )


@app.command()
def run(
    name: str = typer.Option(..., help="實驗組名稱，例如 A_qty"),
    matrix: Path = typer.Option(DEFAULT_MATRIX, help="實驗矩陣 JSON"),
    dry_run: bool = typer.Option(False, help="只列出命令，不實際執行"),
) -> None:
    matrix_path, payload = _load_matrix(matrix)
    experiment = _find_experiment(payload, name)
    _materialize_all(matrix_path, payload)

    root = _matrix_root(matrix_path, payload)
    experiment_root = root / name
    params_path = experiment_root / "train_params.json"
    log_path = experiment_root / "logs" / "train.log"
    run_summary_path = experiment_root / "run_summary.json"

    python_exe = _resolve_python()
    cmd = [
        str(python_exe),
        "-m",
        "src.train_3dgs",
        "--params-json",
        str(params_path.resolve()),
    ]

    console.print(
        Panel(
            f"實驗組: {name}\n"
            f"參數檔: {params_path.resolve()}\n"
            f"輸出: {(experiment_root / '3DGS_models').resolve()}\n"
            f"Log: {log_path.resolve()}\n"
            f"命令: {' '.join(cmd)}",
            title="[bold cyan]準備執行實驗[/]",
            border_style="cyan",
        )
    )

    if dry_run:
        return

    started = time.time()
    started_at = datetime.now().isoformat(timespec="seconds")
    exit_code = _stream_subprocess(cmd, PROJECT_ROOT, log_path)
    finished = time.time()
    finished_at = datetime.now().isoformat(timespec="seconds")

    stats_path = _latest_val_stats(experiment_root / "3DGS_models" / "stats")
    summary = {
        "matrix_name": _matrix_name(matrix_path, payload),
        "experiment_name": name,
        "command": cmd,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": round(finished - started, 3),
        "exit_code": exit_code,
        "log_path": str(log_path.resolve()),
        "latest_val_stats": str(stats_path.resolve()) if stats_path else "",
    }
    _write_json(run_summary_path, summary)

    if exit_code != 0:
        console.print(
            Panel(
                f"實驗 {name} 失敗，exit code={exit_code}\n"
                f"請查看：{log_path.resolve()}",
                title="[bold red]實驗失敗[/]",
                border_style="red",
            )
        )
        raise typer.Exit(exit_code)

    console.print(
        Panel(
            f"實驗 {name} 完成\n"
            f"耗時: {summary['elapsed_seconds']} 秒\n"
            f"最新驗證檔: {summary['latest_val_stats'] or '尚未找到'}",
            title="[bold green]實驗完成[/]",
            border_style="green",
        )
    )


@app.command()
def collect(
    matrix: Path = typer.Option(DEFAULT_MATRIX, help="實驗矩陣 JSON"),
) -> None:
    matrix_path, payload = _load_matrix(matrix)
    root = _matrix_root(matrix_path, payload)
    rows: list[dict[str, Any]] = []

    for experiment in payload.get("experiments", []):
        name = str(experiment["name"])
        experiment_root = root / name
        stats_path = _latest_val_stats(experiment_root / "3DGS_models" / "stats")
        summary_path = experiment_root / "run_summary.json"
        observation = _load_observation(experiment_root)
        metrics: dict[str, Any] = {}
        if stats_path and stats_path.exists():
            metrics = _read_json(stats_path)
        run_summary: dict[str, Any] = {}
        if summary_path.exists():
            run_summary = _read_json(summary_path)

        rows.append(
            {
                "name": name,
                "question": experiment.get("question", ""),
                "comparison_role": experiment.get("comparison_role", ""),
                "psnr": metrics.get("psnr"),
                "ssim": metrics.get("ssim"),
                "lpips": metrics.get("lpips"),
                "num_GS": metrics.get("num_GS"),
                "train_time_sec": run_summary.get("elapsed_seconds"),
                "val_stats": str(stats_path.resolve()) if stats_path else "",
                "unity_subjective": observation.get("unity_subjective", ""),
                "prune_needed": observation.get("prune_needed"),
                "notes": observation.get("notes", ""),
            }
        )

    comparison_json = root / "comparison_report.json"
    comparison_md = root / "comparison_report.md"
    _write_json(
        comparison_json,
        {
            "matrix_name": _matrix_name(matrix_path, payload),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "rows": rows,
        },
    )

    lines = [
        f"# {_matrix_name(matrix_path, payload)} comparison",
        "",
        "| Name | Role | PSNR | SSIM | LPIPS | num_GS | Train Sec | Unity | Prune |",
        "|------|------|------|------|-------|--------|-----------|-------|-------|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["name"]),
                    str(row["comparison_role"]),
                    "" if row["psnr"] is None else f"{row['psnr']:.4f}",
                    "" if row["ssim"] is None else f"{row['ssim']:.4f}",
                    "" if row["lpips"] is None else f"{row['lpips']:.4f}",
                    "" if row["num_GS"] is None else str(row["num_GS"]),
                    "" if row["train_time_sec"] is None else str(row["train_time_sec"]),
                    str(row["unity_subjective"] or ""),
                    "" if row["prune_needed"] is None else str(row["prune_needed"]),
                ]
            )
            + " |"
        )
    comparison_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    console.print(
        Panel(
            f"JSON: {comparison_json.resolve()}\n"
            f"Markdown: {comparison_md.resolve()}",
            title="[bold green]已生成比較報告[/]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    app()
