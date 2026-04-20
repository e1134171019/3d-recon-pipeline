from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import typer

from src.run_mask_route_opencv import _build_machine_roi_mask

app = typer.Typer(help="L0-S1 windowed frame selection baseline.")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMGDIR = PROJECT_ROOT / "data" / "frames_1600"
DEFAULT_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "l0_selection"
_YOLO_MODEL_CACHE: dict[str, Any] = {}


@dataclass
class FrameScore:
    image: str
    order_index: int
    roi_x0: int
    roi_y0: int
    roi_x1: int
    roi_y1: int
    blur_score: float
    feature_count: float
    glare_ratio: float
    duplicate_penalty: float = 0.0
    combined_score: float | None = None
    selected: bool = False
    window_index: int | None = None


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_python() -> Path:
    candidate = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if candidate.exists():
        return candidate
    return Path(shutil.which("python") or "python")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if math.isclose(vmin, vmax):
        return [0.5 for _ in values]
    scale = vmax - vmin
    return [(v - vmin) / scale for v in values]


def _roi_from_mask(mask: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    coords = cv2.findNonZero(mask)
    if coords is None:
        return 0, 0, width, height
    x, y, w, h = cv2.boundingRect(coords)
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width, x + w)
    y1 = min(height, y + h)
    return x0, y0, x1, y1


def _build_heuristic_roi_mask(frame_bgr: np.ndarray) -> np.ndarray:
    roi_mask, _ = _build_machine_roi_mask(
        image_bgr=frame_bgr,
        canny_low=50,
        canny_high=150,
        red_sat_floor=70,
        red_val_floor=45,
        min_component_area_ratio=0.003,
        roi_margin_ratio=0.05,
        fallback_center_x=0.62,
        fallback_width_ratio=0.70,
        fallback_height_ratio=0.94,
    )
    return roi_mask


def _load_yolo_model(model_name_or_path: str) -> Any:
    if model_name_or_path in _YOLO_MODEL_CACHE:
        return _YOLO_MODEL_CACHE[model_name_or_path]

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise SystemExit(
            "semantic ROI mode requires ultralytics. "
            "Install it in .venv or switch --roi-mode heuristic."
        ) from exc

    model = YOLO(model_name_or_path)
    _YOLO_MODEL_CACHE[model_name_or_path] = model
    return model


def _score_mask_candidate(mask: np.ndarray) -> float:
    coords = cv2.findNonZero(mask)
    if coords is None:
        return -1.0

    x, y, w, h = cv2.boundingRect(coords)
    area = float(cv2.countNonZero(mask))
    height, width = mask.shape[:2]
    cx = x + w / 2.0
    cy = y + h / 2.0
    dx = abs(cx - width / 2.0) / max(width / 2.0, 1.0)
    dy = abs(cy - height / 2.0) / max(height / 2.0, 1.0)
    centrality = 1.0 - min(1.0, (dx + dy) / 2.0)
    area_ratio = area / float(width * height)
    if area_ratio < 0.01:
        return -1.0
    return area_ratio * 0.6 + centrality * 0.4


def _build_semantic_roi_mask(
    frame_bgr: np.ndarray,
    yolo_model: Any,
    conf: float,
    margin_ratio: float = 0.03,
) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    prediction = yolo_model.predict(
        source=frame_bgr,
        conf=conf,
        verbose=False,
    )
    if not prediction:
        return _build_heuristic_roi_mask(frame_bgr)

    result = prediction[0]
    masks = getattr(result, "masks", None)
    if masks is None or masks.data is None:
        return _build_heuristic_roi_mask(frame_bgr)

    best_mask: np.ndarray | None = None
    best_score = -1.0
    for tensor_mask in masks.data:
        mask_np = tensor_mask.detach().cpu().numpy()
        if mask_np.ndim != 2:
            continue
        mask_np = (mask_np > 0.5).astype(np.uint8) * 255
        if mask_np.shape != (height, width):
            mask_np = cv2.resize(mask_np, (width, height), interpolation=cv2.INTER_NEAREST)
        score = _score_mask_candidate(mask_np)
        if score > best_score:
            best_score = score
            best_mask = mask_np

    if best_mask is None:
        return _build_heuristic_roi_mask(frame_bgr)

    kernel_size = max(3, int(min(height, width) * margin_ratio) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(best_mask, kernel, iterations=1)
    return dilated


def calculate_frame_score(
    frame: np.ndarray,
    roi_mask: np.ndarray,
    previous_winner: np.ndarray | None,
    use_duplicate_penalty: bool,
) -> tuple[dict[str, float], np.ndarray, tuple[int, int, int, int]]:
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = _roi_from_mask(roi_mask, width, height)
    roi = frame[y0:y1, x0:x1]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    blur_score = float(cv2.Laplacian(roi_gray, cv2.CV_64F).var())

    corners = cv2.goodFeaturesToTrack(
        roi_gray,
        maxCorners=800,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7,
    )
    feature_count = float(0 if corners is None else len(corners))

    v = roi_hsv[:, :, 2]
    s = roi_hsv[:, :, 1]
    glare_mask = (v >= 245) & (s <= 160)
    glare_ratio = float(np.count_nonzero(glare_mask)) / float(glare_mask.size)

    roi_gray_small = cv2.resize(roi_gray, (160, 160), interpolation=cv2.INTER_AREA)
    if not use_duplicate_penalty or previous_winner is None:
        duplicate_penalty = 0.0
    else:
        duplicate_distance = float(np.mean(cv2.absdiff(roi_gray_small, previous_winner)))
        duplicate_penalty = 255.0 - duplicate_distance

    return (
        {
            "blur_score": blur_score,
            "feature_count": feature_count,
            "glare_ratio": glare_ratio,
            "duplicate_penalty": duplicate_penalty,
        },
        roi_gray_small,
        (x0, y0, x1, y1),
    )


def _score_frame(
    image_path: Path,
    order_index: int,
    previous_winner: np.ndarray | None,
    roi_mode: str,
    semantic_model: Any | None,
    semantic_conf: float,
    use_duplicate_penalty: bool,
) -> tuple[FrameScore, np.ndarray]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Failed to read image: {image_path}")

    if roi_mode == "semantic":
        assert semantic_model is not None
        roi_mask = _build_semantic_roi_mask(image, semantic_model, semantic_conf)
    else:
        roi_mask = _build_heuristic_roi_mask(image)
    metrics, roi_gray_small, (x0, y0, x1, y1) = calculate_frame_score(
        image,
        roi_mask,
        previous_winner,
        use_duplicate_penalty,
    )

    score = FrameScore(
        image=image_path.name,
        order_index=order_index,
        roi_x0=x0,
        roi_y0=y0,
        roi_x1=x1,
        roi_y1=y1,
        blur_score=float(metrics["blur_score"]),
        feature_count=float(metrics["feature_count"]),
        glare_ratio=float(metrics["glare_ratio"]),
        duplicate_penalty=float(metrics["duplicate_penalty"]),
    )
    return score, roi_gray_small


def _run_process(cmd: list[str], log_path: Path) -> None:
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    with log_path.open("w", encoding="utf-8", errors="replace", buffering=1) as log_file:
        process = subprocess.Popen(
            cmd,
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
            print(line, end="", flush=True)
            log_file.write(line)
        code = process.wait()
    if code != 0:
        raise SystemExit(code)


def _latest_sparse_dir(work_root: Path) -> Path:
    sparse_root = work_root / "sparse"
    best = sparse_root / "best"
    if best.exists():
        return best
    for candidate in sorted(sparse_root.iterdir()):
        if candidate.is_dir():
            return candidate
    raise SystemExit(f"No sparse model found under {sparse_root}")


@app.command()
def main(
    imgdir: str = typer.Option(str(DEFAULT_IMGDIR), help="Input image directory."),
    run_root: str = typer.Option("", help="Experiment root. Empty = auto timestamped."),
    window_size: int = typer.Option(6, help="Frames per window."),
    keep_top_k: int = typer.Option(1, help="Frames kept per window."),
    roi_mode: str = typer.Option("heuristic", help="ROI mode: heuristic or semantic."),
    semantic_model_path: str = typer.Option("weights/yolo11n-seg.pt", help="YOLO segmentation model path/name for semantic ROI."),
    semantic_conf: float = typer.Option(0.25, help="Confidence threshold for semantic ROI."),
    use_duplicate_penalty: bool = typer.Option(True, help="Apply duplicate penalty in window scoring."),
    run_sfm: bool = typer.Option(True, help="Run baseline SfM on selected frames."),
    min_points3d: int = typer.Option(1, help="SfM Step 3 min points gate for subset validation."),
) -> None:
    if roi_mode not in {"heuristic", "semantic"}:
        raise SystemExit(f"Unsupported roi_mode: {roi_mode}")

    imgdir_path = Path(imgdir)
    if not imgdir_path.is_absolute():
        imgdir_path = (PROJECT_ROOT / imgdir_path).resolve()
    if not imgdir_path.exists():
        raise SystemExit(f"Image directory not found: {imgdir_path}")

    if run_root:
        run_root_path = Path(run_root)
        if not run_root_path.is_absolute():
            run_root_path = (PROJECT_ROOT / run_root_path).resolve()
    else:
        run_root_path = DEFAULT_ROOT / f"L0_S1_windowed_selection_{_timestamp()}"

    selected_dir = run_root_path / "selected_frames"
    reports_dir = run_root_path / "reports"
    logs_dir = run_root_path / "logs"
    work_root = run_root_path / "SfM_models" / "l0_s1_windowed"
    for path in (selected_dir, reports_dir, logs_dir, work_root):
        path.mkdir(parents=True, exist_ok=True)

    images = sorted(imgdir_path.glob("*.jpg"))
    if not images:
        raise SystemExit(f"No JPG images found under: {imgdir_path}")

    print(f"[INFO] L0-S1 source: {imgdir_path}", flush=True)
    print(f"[INFO] window_size={window_size}, keep_top_k={keep_top_k}", flush=True)
    print(f"[INFO] roi_mode={roi_mode}", flush=True)
    print(f"[INFO] use_duplicate_penalty={use_duplicate_penalty}", flush=True)
    print(f"[INFO] run_root={run_root_path}", flush=True)

    semantic_model = None
    if roi_mode == "semantic":
        print(f"[INFO] loading semantic ROI model: {semantic_model_path}", flush=True)
        semantic_model = _load_yolo_model(semantic_model_path)

    frame_scores: list[FrameScore] = []
    selected_names: list[str] = []
    previous_winner: np.ndarray | None = None
    for window_index, start in enumerate(range(0, len(images), window_size)):
        window_images = images[start:start + window_size]
        window: list[FrameScore] = []
        window_smalls: list[np.ndarray] = []
        for offset, image_path in enumerate(window_images):
            score, roi_small = _score_frame(
                image_path,
                start + offset,
                previous_winner,
                roi_mode,
                semantic_model,
                semantic_conf,
                use_duplicate_penalty,
            )
            score.window_index = window_index
            window.append(score)
            window_smalls.append(roi_small)
            frame_scores.append(score)

        blur_norm = _normalize([row.blur_score for row in window])
        feat_norm = _normalize([row.feature_count for row in window])
        if use_duplicate_penalty:
            duplicate_norm = _normalize([row.duplicate_penalty for row in window])
        else:
            duplicate_norm = [0.0 for _ in window]
        glare_norm = _normalize([row.glare_ratio for row in window])

        for row, b, f, d, g in zip(window, blur_norm, feat_norm, duplicate_norm, glare_norm):
            row.combined_score = 0.35 * f + 0.30 * b - 0.25 * d - 0.10 * g

        ranked = sorted(window, key=lambda row: row.combined_score or 0.0, reverse=True)
        winner_names = {row.image for row in ranked[:keep_top_k]}
        for row in window:
            if row.image in winner_names:
                row.selected = True
                selected_names.append(row.image)
        for idx, row in enumerate(ranked[:keep_top_k]):
            row.selected = True
            if idx == 0:
                previous_winner = window_smalls[window.index(row)]

    selected_names_set = set(selected_names)
    for image_path in images:
        if image_path.name in selected_names_set:
            shutil.copy2(image_path, selected_dir / image_path.name)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "route": "l0_selection",
        "variant": "L0_S1_windowed_selection",
        "source_imgdir": str(imgdir_path),
        "selected_imgdir": str(selected_dir),
        "total_images": len(images),
        "window_size": window_size,
        "keep_top_k": keep_top_k,
        "roi_mode": roi_mode,
        "semantic_model_path": semantic_model_path if roi_mode == "semantic" else "",
        "semantic_conf": semantic_conf if roi_mode == "semantic" else None,
        "use_duplicate_penalty": use_duplicate_penalty,
        "selected_count": len(selected_names),
        "selected_examples_head": selected_names[:10],
        "selected_examples_tail": selected_names[-10:],
        "avg_blur_score": float(np.mean([row.blur_score for row in frame_scores])),
        "avg_feature_count": float(np.mean([row.feature_count for row in frame_scores])),
        "avg_glare_ratio": float(np.mean([row.glare_ratio for row in frame_scores])),
        "avg_duplicate_penalty": float(np.mean([row.duplicate_penalty for row in frame_scores])),
    }
    _write_json(reports_dir / "selection_summary.json", summary)
    _write_json(
        reports_dir / "frame_scores.json",
        {"frames": [row.__dict__ for row in frame_scores]},
    )
    _write_json(
        run_root_path / "selection_params.json",
        {
            "imgdir": str(imgdir_path),
            "window_size": window_size,
            "keep_top_k": keep_top_k,
            "roi_mode": roi_mode,
            "semantic_model_path": semantic_model_path if roi_mode == "semantic" else "",
            "semantic_conf": semantic_conf if roi_mode == "semantic" else None,
            "use_duplicate_penalty": use_duplicate_penalty,
            "run_sfm": run_sfm,
            "min_points3d": min_points3d,
        },
    )
    print(
        f"[OK] Selection complete: {len(selected_names)} / {len(images)} frames -> {selected_dir}",
        flush=True,
    )

    if not run_sfm:
        return

    python = str(_resolve_python())
    sfm_params = {
        "sfm_params": {
            "profile_name": run_root_path.name,
            "recommended_params": {
                "imgdir": str(selected_dir.resolve()),
                "work": str(work_root.resolve()),
                "validation_report": str((reports_dir / "pointcloud_validation_report.json").resolve()),
                "matcher": "sequential",
                "mapper_type": "incremental",
                "max_features": 8192,
                "seq_overlap": 10,
                "loop_detection": False,
                "sift_peak_threshold": 0.006666666666666667,
                "sift_edge_threshold": 10,
                "min_points3d": int(min_points3d),
            }
        }
    }
    sfm_params_path = run_root_path / "sfm_params.json"
    _write_json(sfm_params_path, sfm_params)

    cmd = [python, "-u", "-m", "src.sfm_colmap", "--params-json", str(sfm_params_path.resolve())]
    print("[INFO] Running baseline SfM on selected frames...", flush=True)
    _run_process(cmd, logs_dir / "sfm.log")

    report_path = reports_dir / "pointcloud_validation_report.json"
    report_payload: dict[str, Any] | None = None
    sparse_best: Path | None = None
    if report_path.exists():
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        sparse_dir = report_payload.get("sparse_dir")
        if sparse_dir:
            sparse_best = Path(sparse_dir)

    if sparse_best is None:
        sparse_best = _latest_sparse_dir(work_root)

    _write_json(
        run_root_path / "run_summary.json",
        {
            "timestamp": datetime.now().isoformat(),
            "route": "l0_selection",
            "variant": "L0_S1_windowed_selection",
            "selected_count": len(selected_names),
            "selected_imgdir": str(selected_dir.resolve()),
            "sparse_model": str(sparse_best.resolve()),
            "validation_report": str(report_path.resolve()) if report_path.exists() else None,
            "geometry": report_payload,
        },
    )
    print(f"[OK] L0-S1 finished: {run_root_path}", flush=True)


if __name__ == "__main__":
    app()
