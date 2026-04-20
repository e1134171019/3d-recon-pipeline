from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
import time

import cv2
import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel

from src.preprocess_phase0 import get_frame_quality_metrics
from src.run_upstream_matrix import (
    _load_matrix,
    _materialize_all,
    _matrix_name,
    _matrix_root,
)


app = typer.Typer(help="依序執行上游 2x2：Phase 0 / SfM / frozen train baseline")
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATRIX = PROJECT_ROOT / "outputs" / "experiments" / "upstream_2x2_matrix.json"
BASELINE_CLEANED = PROJECT_ROOT / "data" / "frames_cleaned"
BASELINE_1600 = PROJECT_ROOT / "data" / "frames_1600"


def _safe_echo(line: str) -> None:
    text = line.rstrip()
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(safe)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_status(
    status_path: Path,
    matrix_name: str,
    current: str | None,
    completed: list[str],
    failed: list[str],
    status: str = "running",
    stage: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "status": status,
        "matrix_name": matrix_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "current": current,
        "completed": completed,
        "failed": failed,
    }
    if stage is not None:
        payload["stage"] = stage
    if current is None and status != "running":
        payload["finished_at"] = datetime.now().isoformat(timespec="seconds")
    _write_json(status_path, payload)


def _read_json(path: Path) -> dict[str, Any]:
    encodings = ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "cp950", "mbcs")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return json.loads(path.read_text(encoding=encoding))
        except Exception as exc:
            last_error = exc
    raise last_error if last_error is not None else ValueError(f"無法解析 JSON：{path}")


def _resolve_python() -> Path:
    preferred = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    return preferred if preferred.exists() else Path(sys.executable)


def _iter_images(folder: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )


def _clear_dir(folder: Path) -> None:
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)


def _copy_images(src: Path, dst: Path) -> int:
    _clear_dir(dst)
    count = 0
    for path in _iter_images(src):
        shutil.copy2(path, dst / path.name)
        count += 1
    return count


def _resize_to_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _embed_for_similarity(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    flat = small.reshape(-1)
    norm = np.linalg.norm(flat)
    return flat if norm == 0 else flat / norm


def _cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def _load_recommended(path: Path, key: str) -> dict[str, Any]:
    payload = _read_json(path)
    plan = payload.get(key, payload)
    return dict(plan.get("recommended_params", {}))


def _run_process(
    cmd: list[str],
    cwd: Path,
    log_path: Path,
    heartbeat: Callable[[], None] | None = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    last_heartbeat = 0.0
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
            log_file.flush()
            _safe_echo(line)
            if heartbeat is not None:
                now = time.time()
                if now - last_heartbeat >= 5.0:
                    heartbeat()
                    last_heartbeat = now
        if heartbeat is not None:
            heartbeat()
        return process.wait()


def _phase0_complete(experiment_root: Path) -> bool:
    frames_1600_dir = experiment_root / "phase0" / "frames_1600"
    summary = experiment_root / "phase0" / "phase0_summary.json"
    return frames_1600_dir.exists() and any(frames_1600_dir.iterdir()) and summary.exists()


def _run_phase0(experiment_root: Path) -> dict[str, Any]:
    params = _load_recommended(experiment_root / "phase0_params.json", "phase0_params")
    clean_dir = Path(str(params["clean_dir"]))
    frames_1600_dir = Path(str(params["frames_1600_dir"]))
    summary_path = experiment_root / "phase0" / "phase0_summary.json"

    if _phase0_complete(experiment_root):
        return _read_json(summary_path)

    curation_mode = str(params.get("curation_mode", "current"))
    max_side = int(params.get("max_side", 1600))
    blur_threshold = float(params.get("blur_threshold", 120.0))
    brightness_low = float(params.get("brightness_low", 25.0))
    brightness_high = float(params.get("brightness_high", 240.0))
    dedupe_threshold = float(params.get("dedupe_similarity_threshold", 1.0))
    keep_every_n = max(1, int(params.get("keep_every_n", 1)))
    max_frames = int(params.get("max_frames", 0))

    phase0_root = experiment_root / "phase0"
    phase0_root.mkdir(parents=True, exist_ok=True)

    if curation_mode == "current":
        copied_cleaned = _copy_images(BASELINE_CLEANED, clean_dir)
        copied_1600 = _copy_images(BASELINE_1600, frames_1600_dir)
        summary = {
            "mode": "current_copy",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "source_cleaned": str(BASELINE_CLEANED.resolve()),
            "source_frames_1600": str(BASELINE_1600.resolve()),
            "copied_cleaned": copied_cleaned,
            "copied_frames_1600": copied_1600,
            "kept_frames": copied_1600,
            "rejected_blur": 0,
            "rejected_brightness": 0,
            "rejected_duplicate": 0,
        }
        _write_json(summary_path, summary)
        return summary

    _clear_dir(clean_dir)
    _clear_dir(frames_1600_dir)

    source_dir = Path(str(params.get("source_images") or BASELINE_CLEANED))
    if not source_dir.exists():
        raise FileNotFoundError(f"找不到 curated source images：{source_dir}")
    pre_dedupe_frames = len(_iter_images(source_dir))

    kept = 0
    rejected_blur = 0
    rejected_brightness = 0
    rejected_duplicate = 0
    last_embedding: np.ndarray | None = None

    for idx, path in enumerate(_iter_images(source_dir)):
        if keep_every_n > 1 and (idx % keep_every_n) != 0:
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue

        metrics = get_frame_quality_metrics(image)
        if metrics["laplacian_var"] < blur_threshold:
            rejected_blur += 1
            continue
        if metrics["mean_brightness"] < brightness_low or metrics["mean_brightness"] > brightness_high:
            rejected_brightness += 1
            continue

        embedding = _embed_for_similarity(image)
        similarity = _cosine_similarity(last_embedding, embedding)
        if kept > 0 and similarity >= dedupe_threshold:
            rejected_duplicate += 1
            continue

        clean_path = clean_dir / f"frame_{kept:06d}.jpg"
        frame_1600_path = frames_1600_dir / clean_path.name
        cv2.imwrite(str(clean_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        resized = _resize_to_max_side(image, max_side)
        cv2.imwrite(str(frame_1600_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        kept += 1
        last_embedding = embedding

        if max_frames > 0 and kept >= max_frames:
            break

    summary = {
        "mode": "curated",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source_images": str(source_dir.resolve()),
        "input_frames": pre_dedupe_frames,
        "kept_frames": kept,
        "rejected_blur": rejected_blur,
        "rejected_brightness": rejected_brightness,
        "rejected_duplicate": rejected_duplicate,
        "blur_threshold": blur_threshold,
        "brightness_low": brightness_low,
        "brightness_high": brightness_high,
        "dedupe_similarity_threshold": dedupe_threshold,
        "keep_every_n": keep_every_n,
        "max_side": max_side,
    }
    _write_json(summary_path, summary)
    return summary


def _sfm_complete(experiment_root: Path) -> bool:
    report = experiment_root / "reports" / "pointcloud_validation_report.json"
    return report.exists()


def _run_sfm(experiment_root: Path, heartbeat: Callable[[], None] | None = None) -> tuple[int, Path]:
    params_path = experiment_root / "sfm_params.json"
    log_path = experiment_root / "logs" / "sfm.log"
    if _sfm_complete(experiment_root):
        return 0, log_path
    sfm_params = _load_recommended(params_path, "sfm_params")
    work_dir = Path(str(sfm_params["work"]))
    report_path = Path(str(sfm_params["validation_report"]))
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    if report_path.exists():
        report_path.unlink(missing_ok=True)
    cmd = [
        str(_resolve_python()),
        "-m",
        "src.sfm_colmap",
        "--params-json",
        str(params_path.resolve()),
    ]
    return _run_process(cmd, PROJECT_ROOT, log_path, heartbeat=heartbeat), log_path


def _train_complete(experiment_root: Path) -> bool:
    params = _load_recommended(experiment_root / "train_params.json", "train_params")
    iterations = int(params.get("iterations", 30000))
    ckpt = experiment_root / "3DGS_models" / "ckpts" / f"ckpt_{iterations - 1}_rank0.pt"
    return ckpt.exists()


def _run_train(experiment_root: Path, heartbeat: Callable[[], None] | None = None) -> tuple[int, Path]:
    params_path = experiment_root / "train_params.json"
    log_path = experiment_root / "logs" / "train.log"
    if _train_complete(experiment_root):
        return 0, log_path
    train_params = _load_recommended(params_path, "train_params")
    out_dir = Path(str(train_params["outdir"]))
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    cmd = [
        str(_resolve_python()),
        "-m",
        "src.train_3dgs",
        "--params-json",
        str(params_path.resolve()),
    ]
    return _run_process(cmd, PROJECT_ROOT, log_path, heartbeat=heartbeat), log_path


def _latest_val_stats(stats_dir: Path) -> Path | None:
    latest: tuple[int, Path] | None = None
    if not stats_dir.exists():
        return None
    for path in stats_dir.glob("val_step*.json"):
        stem = path.stem.replace("val_step", "")
        if not stem.isdigit():
            continue
        value = int(stem)
        if latest is None or value > latest[0]:
            latest = (value, path)
    return latest[1] if latest else None


def _summarize_experiment(experiment_root: Path) -> dict[str, Any]:
    phase0_summary = experiment_root / "phase0" / "phase0_summary.json"
    sfm_report = experiment_root / "reports" / "pointcloud_validation_report.json"
    train_stats = _latest_val_stats(experiment_root / "3DGS_models" / "stats")
    payload: dict[str, Any] = {
        "experiment_root": str(experiment_root.resolve()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if phase0_summary.exists():
        payload["phase0"] = _read_json(phase0_summary)
    if sfm_report.exists():
        payload["sfm"] = _read_json(sfm_report)
    if train_stats and train_stats.exists():
        payload["train"] = _read_json(train_stats)
        payload["train"]["stats_file"] = str(train_stats.resolve())
    return payload


@app.command()
def run(
    matrix: Path = typer.Option(DEFAULT_MATRIX, help="上游 2x2 矩陣 JSON"),
) -> None:
    matrix_path, payload = _load_matrix(matrix)
    _materialize_all(matrix_path, payload)
    root = _matrix_root(matrix_path, payload)
    suite_log_root = root / "suite_logs"
    suite_log_root.mkdir(parents=True, exist_ok=True)
    status_path = suite_log_root / "upstream_suite_status.json"

    sequence = [str(exp["name"]) for exp in payload.get("experiments", [])]
    completed: list[str] = []
    failed: list[str] = []
    matrix_name = _matrix_name(matrix_path, payload)

    _write_json(
        status_path,
        {
            "status": "running",
            "matrix_name": matrix_name,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "sequence": sequence,
            "current": None,
            "completed": completed,
            "failed": failed,
            "stage": None,
        },
    )

    for name in sequence:
        experiment_root = root / name
        _write_status(status_path, matrix_name, name, completed, failed, "running", stage="phase0")

        try:
            console.print(Panel(f"開始實驗：{name}", title="[bold cyan]Upstream 2x2[/]", border_style="cyan"))
            phase0_summary = _run_phase0(experiment_root)
            _write_json(experiment_root / "phase0" / "phase0_summary.json", phase0_summary)

            def _heartbeat(stage: str) -> Callable[[], None]:
                return lambda: _write_status(status_path, matrix_name, name, completed, failed, "running", stage=stage)

            sfm_code, _ = _run_sfm(experiment_root, heartbeat=_heartbeat("sfm"))
            if sfm_code != 0:
                failed.append(name)
                _write_status(status_path, matrix_name, name, completed, failed, "running", stage="sfm_failed")
                continue

            train_code, _ = _run_train(experiment_root, heartbeat=_heartbeat("train"))
            if train_code != 0:
                failed.append(name)
                _write_status(status_path, matrix_name, name, completed, failed, "running", stage="train_failed")
                continue

            _write_json(experiment_root / "run_summary.json", _summarize_experiment(experiment_root))
            completed.append(name)
            _write_status(status_path, matrix_name, name, completed, failed, "running", stage="completed")
        except Exception as exc:
            failed.append(name)
            _write_json(
                experiment_root / "run_summary.json",
                {
                    "experiment_root": str(experiment_root.resolve()),
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "status": "failed",
                    "error": str(exc),
                },
            )
            _write_status(status_path, matrix_name, name, completed, failed, "running", stage="exception")

    final_status = "completed" if not failed else "completed_with_failures"
    _write_status(status_path, matrix_name, None, completed, failed, final_status, stage=None)

    console.print(
        Panel(
            f"完成組別: {', '.join(completed) if completed else '無'}\n"
            f"失敗組別: {', '.join(failed) if failed else '無'}\n"
            f"狀態檔: {status_path.resolve()}",
            title="[bold green]Upstream 2x2 完成[/]" if not failed else "[bold yellow]Upstream 2x2 結束[/]",
            border_style="green" if not failed else "yellow",
        )
    )


if __name__ == "__main__":
    app()
