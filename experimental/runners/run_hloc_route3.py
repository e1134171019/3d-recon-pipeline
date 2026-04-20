from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pycolmap
import typer
from hloc import extract_features, match_features, reconstruction

from src.sfm_colmap import check_reconstruction, export_signals

app = typer.Typer(help="Route 3 runner for hloc feature/matcher ablations.")


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _slugify(value: str) -> str:
    return (
        value.lower()
        .replace("+", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def _resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def _to_project_relative(project_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _list_images(imgdir: Path) -> list[str]:
    names = [
        p.name
        for p in sorted(imgdir.iterdir())
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if not names:
        raise SystemExit(f"No images found in {imgdir}")
    return names


def _write_sequential_pairs(image_names: list[str], pair_path: Path, overlap: int) -> int:
    pair_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with pair_path.open("w", encoding="utf-8", newline="\n") as f:
        total = len(image_names)
        for i, name in enumerate(image_names):
            upper = min(total, i + overlap + 1)
            for j in range(i + 1, upper):
                f.write(f"{name} {image_names[j]}\n")
                count += 1
    return count


def _latest_val_stats(stats_dir: Path) -> Path | None:
    best_step = -1
    best_path: Path | None = None
    for path in stats_dir.glob("val_step*.json"):
        stem = path.stem
        try:
            step = int(stem.replace("val_step", ""))
        except ValueError:
            continue
        if step > best_step:
            best_step = step
            best_path = path
    return best_path


def _write_run_summary(
    run_root: Path,
    route_name: str,
    feature_conf_name: str,
    match_conf_name: str,
    image_count: int,
    pair_count: int,
    pair_overlap: int,
    sparse_dir: Path,
    train_outdir: Path,
    val_stats_path: Path | None,
) -> None:
    summary = {
        "timestamp": datetime.now().isoformat(),
        "route": route_name,
        "feature_conf": feature_conf_name,
        "match_conf": match_conf_name,
        "image_count": image_count,
        "pair_overlap": pair_overlap,
        "pair_count": pair_count,
        "sparse_dir": str(sparse_dir.resolve()),
        "train_outdir": str(train_outdir.resolve()),
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

    (run_root / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _run_train(
    project_root: Path,
    imgdir: Path,
    sparse_dir: Path,
    train_outdir: Path,
    validation_report: Path,
    iterations: int,
    pair_overlap: int,
) -> None:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.train_3dgs",
        "--imgdir",
        _to_project_relative(project_root, imgdir),
        "--colmap",
        _to_project_relative(project_root, sparse_dir),
        "--outdir",
        _to_project_relative(project_root, train_outdir),
        "--validation-report",
        _to_project_relative(project_root, validation_report),
        "--iterations",
        str(iterations),
        "--sh-degree",
        "3",
        "--densify-until",
        "15000",
        "--grow-grad2d",
        "0.0008",
        "--eval-steps",
        "1000",
    ]
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    print(f"[INFO] Starting 3DGS training (overlap={pair_overlap})", flush=True)
    print(f">> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(project_root), env=env)


@app.command()
def main(
    imgdir: str = typer.Option("data/frames_1600", help="Input image directory."),
    run_root: str = typer.Option("", help="Experiment root. Empty = auto timestamped."),
    pair_overlap: int = typer.Option(10, help="Sequential pair overlap."),
    iterations: int = typer.Option(30000, help="3DGS training iterations."),
    feature_conf: str = typer.Option("superpoint_max", help="hloc feature extractor config name."),
    match_conf: str = typer.Option("superpoint+lightglue", help="hloc matcher config name."),
    skip_train: bool = typer.Option(False, help="Stop after sparse reconstruction."),
) -> None:
    project_root = _project_root()
    imgdir_path = _resolve_project_path(project_root, imgdir)
    if not imgdir_path.exists():
        raise SystemExit(f"Image directory not found: {imgdir_path}")

    if feature_conf not in extract_features.confs:
        available = ", ".join(sorted(extract_features.confs))
        raise SystemExit(f"Unknown feature_conf '{feature_conf}'. Available: {available}")
    if match_conf not in match_features.confs:
        available = ", ".join(sorted(match_features.confs))
        raise SystemExit(f"Unknown match_conf '{match_conf}'. Available: {available}")

    route_name = f"hloc_{_slugify(feature_conf)}_{_slugify(match_conf)}"

    if run_root:
        run_root_path = _resolve_project_path(project_root, run_root)
    else:
        run_root_path = project_root / "outputs" / "experiments" / f"route3_{route_name}_{_timestamp()}"

    hloc_root = run_root_path / "hloc"
    logs_dir = run_root_path / "logs"
    reports_dir = run_root_path / "reports"
    sparse_dir = run_root_path / "SfM_models" / route_name / "sparse" / "0"
    train_outdir = run_root_path / "3DGS_models"

    for path in (hloc_root, logs_dir, reports_dir, sparse_dir, train_outdir):
        path.mkdir(parents=True, exist_ok=True)

    image_names = _list_images(imgdir_path)
    pairs_path = hloc_root / f"pairs-sequential-o{pair_overlap}.txt"
    pair_count = _write_sequential_pairs(image_names, pairs_path, pair_overlap)

    params = {
        "route": route_name,
        "imgdir": _to_project_relative(project_root, imgdir_path),
        "image_count": len(image_names),
        "pair_overlap": pair_overlap,
        "pair_count": pair_count,
        "feature_conf": feature_conf,
        "match_conf": match_conf,
        "skip_train": skip_train,
        "iterations": iterations,
    }
    (run_root_path / "route3_params.json").write_text(
        json.dumps(params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 72, flush=True)
    print(f"Route 3 runner: {route_name}", flush=True)
    print(f"Run root   : {run_root_path}", flush=True)
    print(f"Images     : {imgdir_path} ({len(image_names)} files)", flush=True)
    print(f"Pairs      : {pairs_path} ({pair_count} pairs, overlap={pair_overlap})", flush=True)
    print(f"Sparse dir : {sparse_dir}", flush=True)
    print("=" * 72, flush=True)

    feature_conf_dict = extract_features.confs[feature_conf]
    match_conf_dict = match_features.confs[match_conf]

    print(f"[INFO] Extracting features with {feature_conf}...", flush=True)
    features_path = extract_features.main(
        feature_conf_dict,
        imgdir_path,
        export_dir=hloc_root,
        as_half=True,
        overwrite=False,
    )
    print(f"[OK] Features: {features_path}", flush=True)

    print(f"[INFO] Matching with {match_conf}...", flush=True)
    matches_path = hloc_root / f'{feature_conf_dict["output"]}_{match_conf_dict["output"]}_{pairs_path.stem}.h5'
    matches_path = match_features.main(
        match_conf_dict,
        pairs_path,
        features=features_path,
        export_dir=hloc_root,
        matches=matches_path,
        overwrite=False,
    )
    print(f"[OK] Matches: {matches_path}", flush=True)

    print("[INFO] Running hloc reconstruction...", flush=True)
    reconstruction.main(
        sparse_dir,
        imgdir_path,
        pairs_path,
        features_path,
        matches_path,
        camera_mode=pycolmap.CameraMode.AUTO,
        verbose=True,
    )
    print("[OK] Reconstruction completed.", flush=True)

    result = check_reconstruction(str(sparse_dir))
    export_signals(result, str(sparse_dir.resolve()), reports_dir)
    if not result["pass"] or not result["can_proceed_to_3dgs"]:
        raise SystemExit("Route 3 sparse reconstruction did not pass validation.")

    if skip_train:
        _write_run_summary(run_root_path, len(image_names), pair_count, pair_overlap, sparse_dir, train_outdir, None)
        print("[INFO] skip_train=True, stopping after sparse reconstruction.", flush=True)
        return

    validation_report = reports_dir / "pointcloud_validation_report.json"
    _run_train(
        project_root=project_root,
        imgdir=imgdir_path,
        sparse_dir=sparse_dir,
        train_outdir=train_outdir,
        validation_report=validation_report,
        iterations=iterations,
        pair_overlap=pair_overlap,
    )

    stats_path = _latest_val_stats(train_outdir / "stats")
    _write_run_summary(
        run_root=run_root_path,
        route_name=route_name,
        feature_conf_name=feature_conf,
        match_conf_name=match_conf,
        image_count=len(image_names),
        pair_count=pair_count,
        pair_overlap=pair_overlap,
        sparse_dir=sparse_dir,
        train_outdir=train_outdir,
        val_stats_path=stats_path,
    )
    print(f"[OK] Route 3 run finished: {route_name}", flush=True)


if __name__ == "__main__":
    app()
