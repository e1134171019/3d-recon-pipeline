from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = PROJECT_ROOT / "data" / "viode" / "hub.mp4"
DEFAULT_ROOT = PROJECT_ROOT / "outputs" / "runs"

app = typer.Typer(help="以 ffmpeg Phase 0 v2 從原始影片重跑：Phase0 -> SfM -> 3DGS")
console = Console()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_python() -> Path:
    preferred = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    return preferred if preferred.exists() else Path(sys.executable)


def _run_process(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            text = line.rstrip()
            try:
                print(text)
            except UnicodeEncodeError:
                encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
                safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
                print(safe)
            log_file.write(line)
            log_file.flush()
        return process.wait()


def _build_phase0_params(run_root: Path, source_video: Path) -> Path:
    payload = {
        "phase0_params": {
            "profile_name": run_root.name,
            "recommended_params": {
                "source_video": str(source_video.resolve()),
                "candidate_dir": str((run_root / "phase0" / "candidates").resolve()),
                "clean_dir": str((run_root / "phase0" / "frames_cleaned").resolve()),
                "frames_1600_dir": str((run_root / "phase0" / "frames_1600").resolve()),
                "fps_extract": 12.0,
                "max_side": 1600,
                "blur_threshold": 40.0,
                "brightness_low": 25.0,
                "brightness_high": 240.0,
                "dedupe_similarity_threshold": 0.985,
                "keep_every_n": 1,
                "max_frames": 0,
                "use_hwaccel": True,
                "hwaccel_backend": "auto",
            },
        }
    }
    path = run_root / "phase0_params.json"
    _write_json(path, payload)
    return path


def _build_sfm_params(run_root: Path) -> Path:
    payload = {
        "sfm_params": {
            "profile_name": run_root.name,
            "recommended_params": {
                "work": str((run_root / "SfM_models" / "sift").resolve()),
                "validation_report": str((run_root / "reports" / "pointcloud_validation_report.json").resolve()),
                "imgdir": str((run_root / "phase0" / "frames_1600").resolve()),
                "mapper_type": "incremental",
                "matcher": "sequential",
                "max_features": 8192,
                "seq_overlap": 10,
                "loop_detection": False,
                "sift_peak_threshold": 0.006666666666666667,
                "sift_edge_threshold": 10,
            },
        }
    }
    path = run_root / "sfm_params.json"
    _write_json(path, payload)
    return path


def _build_train_params(run_root: Path) -> Path:
    payload = {
        "train_params": {
            "profile_name": run_root.name,
            "recommended_params": {
                "imgdir": str((run_root / "phase0" / "frames_1600").resolve()),
                "colmap": str((run_root / "SfM_models" / "sift" / "sparse" / "best").resolve()),
                "outdir": str((run_root / "3DGS_models").resolve()),
                "validation_report": str((run_root / "reports" / "pointcloud_validation_report.json").resolve()),
                "iterations": 30000,
                "sh_degree": 3,
                "densify_until": 15000,
                "scene_scale": 0.0,
                "scale_json": "",
                "eval_steps": 1000,
                "data_factor": 1,
                "absgrad": False,
                "grow_grad2d": 0.0008,
                "antialiased": False,
                "random_bkgd": False,
            },
        }
    }
    path = run_root / "train_params.json"
    _write_json(path, payload)
    return path


@app.command()
def run(
    source_video: str = typer.Option(str(DEFAULT_VIDEO), help="原始影片"),
    run_name: str = typer.Option("", help="可選 run name"),
    root_dir: str = typer.Option(str(DEFAULT_ROOT), help="runs 根目錄"),
) -> None:
    source_video_path = Path(source_video)
    if not source_video_path.is_absolute():
        source_video_path = (PROJECT_ROOT / source_video_path).resolve()
    if not source_video_path.exists():
        raise typer.BadParameter(f"找不到影片：{source_video_path}")

    root_path = Path(root_dir)
    if not root_path.is_absolute():
        root_path = (PROJECT_ROOT / root_path).resolve()
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root_path / (run_name or f"ffmpeg_fullchain_{run_stamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    phase0_params = _build_phase0_params(run_dir, source_video_path)
    sfm_params = _build_sfm_params(run_dir)
    train_params = _build_train_params(run_dir)

    console.print(
        Panel.fit(
            f"[bold cyan]ffmpeg full-chain[/]\n"
            f"run = {run_dir.name}\n"
            f"video = {source_video_path}\n"
            f"Phase0 -> SfM -> 3DGS",
            title="Start",
        )
    )

    python = str(_resolve_python())

    phase0_cmd = [python, "-m", "src.phase0_runner_v2", "--params-json", str(phase0_params.resolve())]
    sfm_cmd = [python, "-m", "src.sfm_colmap", "--params-json", str(sfm_params.resolve())]
    train_cmd = [python, "-m", "src.train_3dgs", "--params-json", str(train_params.resolve())]

    rc = _run_process(phase0_cmd, run_dir / "logs" / "phase0.log")
    if rc != 0:
        raise typer.Exit(rc)
    rc = _run_process(sfm_cmd, run_dir / "logs" / "sfm.log")
    if rc != 0:
        raise typer.Exit(rc)
    rc = _run_process(train_cmd, run_dir / "logs" / "train.log")
    if rc != 0:
        raise typer.Exit(rc)

    console.print(f"[green]full-chain 完成[/] {run_dir}")


if __name__ == "__main__":
    app()
