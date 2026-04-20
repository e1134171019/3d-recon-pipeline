from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from PIL import Image, ImageOps

app = typer.Typer(help="Data-route runner: static photos -> baseline SfM -> baseline 3DGS.")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "photos_static_round1_raw"
DEFAULT_WORK_DIR = PROJECT_ROOT / "data" / "photos_static_round1_work"
DEFAULT_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "data_route"
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_python() -> Path:
    preferred = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    return preferred if preferred.exists() else Path(sys.executable)


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _list_images(imgdir: Path) -> list[Path]:
    images = [
        path
        for path in sorted(imgdir.iterdir())
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    ]
    if not images:
        raise SystemExit(f"No images found in {imgdir}")
    return images


def _prepare_working_set(
    raw_dir: Path,
    work_dir: Path,
    run_dir: Path,
    max_side: int,
    overwrite_work: bool,
) -> tuple[list[dict[str, Any]], Path]:
    if overwrite_work and work_dir.exists():
        for child in work_dir.iterdir():
            if child.is_file():
                child.unlink()

    work_dir.mkdir(parents=True, exist_ok=True)
    images = _list_images(raw_dir)
    manifest: list[dict[str, Any]] = []

    max_side = max(1, int(max_side))
    if hasattr(Image, "Resampling"):
        lanczos = Image.Resampling.LANCZOS
    else:
        lanczos = Image.LANCZOS

    for index, source in enumerate(images, start=1):
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            src_w, src_h = image.size
            scale = min(1.0, float(max_side) / float(max(src_w, src_h)))
            if scale < 1.0:
                dst_w = max(1, int(round(src_w * scale)))
                dst_h = max(1, int(round(src_h * scale)))
                image = image.resize((dst_w, dst_h), lanczos)
            else:
                dst_w, dst_h = src_w, src_h

            dst_name = f"photo_{index:06d}.jpg"
            dst_path = work_dir / dst_name
            image.save(dst_path, format="JPEG", quality=95, subsampling=0)

        manifest.append(
            {
                "index": index,
                "source": str(source.resolve()),
                "work_image": str(dst_path.resolve()),
                "source_size": [src_w, src_h],
                "work_size": [dst_w, dst_h],
                "scale": round(scale, 6),
            }
        )

    manifest_path = run_dir / "prep" / "photo_manifest.json"
    _write_json(
        manifest_path,
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "raw_dir": str(raw_dir.resolve()),
            "work_dir": str(work_dir.resolve()),
            "image_count": len(manifest),
            "max_side": max_side,
            "photos": manifest,
        },
    )
    return manifest, manifest_path


def _run_process(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.pop("TORCH_CUDA_ARCH_LIST", None)

    print(">> " + " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8", errors="replace", buffering=1) as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
        rc = process.wait()
    if rc != 0:
        raise SystemExit(rc)


def _latest_val_stats(stats_dir: Path) -> Path | None:
    best_step = -1
    best_path: Path | None = None
    for path in stats_dir.glob("val_step*.json"):
        try:
            step = int(path.stem.replace("val_step", ""))
        except ValueError:
            continue
        if step > best_step:
            best_step = step
            best_path = path
    return best_path


def _build_sfm_params(run_dir: Path, imgdir: Path) -> Path:
    path = run_dir / "sfm_params.json"
    _write_json(
        path,
        {
            "sfm_params": {
                "profile_name": run_dir.name,
                "recommended_params": {
                    "work": str((run_dir / "SfM_models" / "sift").resolve()),
                    "validation_report": str(
                        (run_dir / "reports" / "pointcloud_validation_report.json").resolve()
                    ),
                    "imgdir": str(imgdir.resolve()),
                    "mapper_type": "incremental",
                    "matcher": "sequential",
                    "max_features": 8192,
                    "seq_overlap": 10,
                    "loop_detection": False,
                    "sift_peak_threshold": 0.006666666666666667,
                    "sift_edge_threshold": 10,
                },
            }
        },
    )
    return path


def _build_train_params(run_dir: Path, imgdir: Path) -> Path:
    path = run_dir / "train_params.json"
    _write_json(
        path,
        {
            "train_params": {
                "profile_name": run_dir.name,
                "recommended_params": {
                    "imgdir": str(imgdir.resolve()),
                    "colmap": str((run_dir / "SfM_models" / "sift" / "sparse" / "best").resolve()),
                    "outdir": str((run_dir / "3DGS_models").resolve()),
                    "validation_report": str(
                        (run_dir / "reports" / "pointcloud_validation_report.json").resolve()
                    ),
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
        },
    )
    return path


def _write_run_summary(
    run_dir: Path,
    raw_dir: Path,
    work_dir: Path,
    image_count: int,
    manifest_path: Path,
    val_stats_path: Path | None,
) -> None:
    summary: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "route": "data_route",
        "variant": "D_base_photo",
        "raw_dir": str(raw_dir.resolve()),
        "work_dir": str(work_dir.resolve()),
        "image_count": image_count,
        "manifest": str(manifest_path.resolve()),
        "metrics": None,
    }

    if val_stats_path and val_stats_path.exists():
        metrics = json.loads(val_stats_path.read_text(encoding="utf-8"))
        summary["metrics"] = {
            "psnr": metrics.get("psnr"),
            "ssim": metrics.get("ssim"),
            "lpips": metrics.get("lpips"),
            "num_GS": metrics.get("num_GS"),
            "stats_file": str(val_stats_path.resolve()),
        }

    _write_json(run_dir / "run_summary.json", summary)


@app.command()
def main(
    raw_dir: str = typer.Option(str(DEFAULT_RAW_DIR), help="Raw static photos directory."),
    work_dir: str = typer.Option(str(DEFAULT_WORK_DIR), help="Prepared working photo directory."),
    run_name: str = typer.Option("", help="Optional run name."),
    root_dir: str = typer.Option(str(DEFAULT_ROOT), help="Experiment root directory."),
    max_side: int = typer.Option(1600, help="Long-side limit for working images."),
    overwrite_work: bool = typer.Option(True, help="Clear and rebuild working directory."),
    skip_prepare: bool = typer.Option(False, help="Reuse existing work_dir and skip photo preparation."),
) -> None:
    raw_dir_path = _resolve_project_path(raw_dir)
    work_dir_path = _resolve_project_path(work_dir)
    root_dir_path = _resolve_project_path(root_dir)

    if not raw_dir_path.exists():
        raise SystemExit(f"Raw photo directory not found: {raw_dir_path}")

    run_dir = root_dir_path / (run_name or f"D_base_photo_{_timestamp()}")
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72, flush=True)
    print("Data Route runner: D_base_photo", flush=True)
    print(f"Run root : {run_dir}", flush=True)
    print(f"Raw dir  : {raw_dir_path}", flush=True)
    print(f"Work dir : {work_dir_path}", flush=True)
    print("=" * 72, flush=True)

    if skip_prepare:
        work_images = _list_images(work_dir_path)
        manifest_path = run_dir / "prep" / "photo_manifest.json"
        _write_json(
            manifest_path,
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "raw_dir": str(raw_dir_path.resolve()),
                "work_dir": str(work_dir_path.resolve()),
                "image_count": len(work_images),
                "max_side": max_side,
                "mode": "reuse_existing_work_dir",
            },
        )
        image_count = len(work_images)
    else:
        manifest, manifest_path = _prepare_working_set(
            raw_dir=raw_dir_path,
            work_dir=work_dir_path,
            run_dir=run_dir,
            max_side=max_side,
            overwrite_work=overwrite_work,
        )
        image_count = len(manifest)
        print(f"[OK] Prepared working set: {image_count} photos", flush=True)

    route_params = {
        "route": "data_route",
        "variant": "D_base_photo",
        "raw_dir": _project_relative(raw_dir_path),
        "work_dir": _project_relative(work_dir_path),
        "image_count": image_count,
        "max_side": max_side,
        "skip_prepare": skip_prepare,
        "overwrite_work": overwrite_work,
        "sfm_baseline": "U_base",
        "train_baseline": "grow_grad2d=0.0008",
    }
    _write_json(run_dir / "data_route_params.json", route_params)

    sfm_params = _build_sfm_params(run_dir, work_dir_path)
    train_params = _build_train_params(run_dir, work_dir_path)

    python = str(_resolve_python())
    sfm_cmd = [python, "-u", "-m", "src.sfm_colmap", "--params-json", str(sfm_params.resolve())]
    train_cmd = [python, "-u", "-m", "src.train_3dgs", "--params-json", str(train_params.resolve())]

    _run_process(sfm_cmd, run_dir / "logs" / "sfm.log")
    _run_process(train_cmd, run_dir / "logs" / "train.log")

    stats_path = _latest_val_stats(run_dir / "3DGS_models" / "stats")
    _write_run_summary(
        run_dir=run_dir,
        raw_dir=raw_dir_path,
        work_dir=work_dir_path,
        image_count=image_count,
        manifest_path=manifest_path,
        val_stats_path=stats_path,
    )
    print(f"[OK] Data route run finished: {run_dir.name}", flush=True)


if __name__ == "__main__":
    app()
