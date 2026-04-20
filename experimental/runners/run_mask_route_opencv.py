from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import typer
from tqdm import tqdm

app = typer.Typer(help="Mask Route A experiment runner: NumPy + OpenCV adaptive masking baseline.")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMGDIR = PROJECT_ROOT / "data" / "frames_1600"
DEFAULT_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "mask_route_a"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_python() -> Path:
    preferred = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    return preferred if preferred.exists() else Path(sys.executable)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _list_images(imgdir: Path) -> list[Path]:
    images = [
        path
        for path in sorted(imgdir.iterdir())
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if not images:
        raise SystemExit(f"No images found in {imgdir}")
    return images


def _run_process(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

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


def _build_highlight_mask(
    image_bgr: np.ndarray,
    highlight_percentile: float,
    brightness_floor: int,
    saturation_ceiling: int,
    min_blob_area_ratio: float,
    dilation_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    l = lab[:, :, 0]

    v_threshold = int(max(brightness_floor, np.percentile(v, highlight_percentile)))
    l_threshold = int(max(brightness_floor, np.percentile(l, highlight_percentile)))

    bright_mask = (v >= v_threshold) | (l >= l_threshold)
    low_sat_mask = s <= saturation_ceiling
    raw_mask = np.where(bright_mask & low_sat_mask, 255, 0).astype(np.uint8)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, open_kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)

    height, width = cleaned.shape
    min_blob_area = max(32, int(height * width * min_blob_area_ratio))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    filtered = np.zeros_like(cleaned)
    kept_components = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_blob_area:
            continue
        filtered[labels == label] = 255
        kept_components += 1

    if dilation_size > 1:
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_size, dilation_size)
        )
        filtered = cv2.dilate(filtered, dilate_kernel, iterations=1)

    mask_ratio = float(np.count_nonzero(filtered)) / float(filtered.size)
    stats_payload = {
        "v_threshold": v_threshold,
        "l_threshold": l_threshold,
        "kept_components": kept_components,
        "mask_ratio": mask_ratio,
    }
    return filtered, stats_payload


def _build_machine_roi_mask(
    image_bgr: np.ndarray,
    canny_low: int,
    canny_high: int,
    red_sat_floor: int,
    red_val_floor: int,
    min_component_area_ratio: float,
    roi_margin_ratio: float,
    fallback_center_x: float,
    fallback_width_ratio: float,
    fallback_height_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, edge_kernel, iterations=1)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    red_mask_1 = (h <= 10) & (s >= red_sat_floor) & (v >= red_val_floor)
    red_mask_2 = (h >= 170) & (s >= red_sat_floor) & (v >= red_val_floor)
    red_mask = np.where(red_mask_1 | red_mask_2, 255, 0).astype(np.uint8)

    evidence = cv2.bitwise_or(edges, red_mask)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    evidence = cv2.morphologyEx(evidence, cv2.MORPH_CLOSE, close_kernel)
    evidence = cv2.dilate(evidence, close_kernel, iterations=1)

    min_component_area = max(128, int(height * width * min_component_area_ratio))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(evidence, connectivity=8)

    best_score = float("-inf")
    best_bbox: tuple[int, int, int, int] | None = None
    best_component_area = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h_box = int(stats[label, cv2.CC_STAT_HEIGHT])
        component = labels == label
        red_support = int(np.count_nonzero((red_mask > 0) & component))
        edge_support = int(np.count_nonzero((edges > 0) & component))

        center_x = (x + w / 2.0) / float(width)
        center_y = (y + h_box / 2.0) / float(height)
        center_penalty = abs(center_x - fallback_center_x) + abs(center_y - 0.5)

        score = (
            area / float(height * width)
            + red_support / 5000.0
            + edge_support / 20000.0
            - center_penalty * 0.35
        )
        if score > best_score:
            best_score = score
            best_bbox = (x, y, w, h_box)
            best_component_area = area

    used_fallback = best_bbox is None
    if best_bbox is None:
        fallback_w = int(width * fallback_width_ratio)
        fallback_h = int(height * fallback_height_ratio)
        center_x_px = int(width * fallback_center_x)
        x0 = max(0, center_x_px - fallback_w // 2)
        y0 = max(0, int(height * 0.5) - fallback_h // 2)
        x1 = min(width, x0 + fallback_w)
        y1 = min(height, y0 + fallback_h)
    else:
        x, y, w, h_box = best_bbox
        margin_x = int(width * roi_margin_ratio)
        margin_y = int(height * roi_margin_ratio)
        x0 = max(0, x - margin_x)
        y0 = max(0, y - margin_y)
        x1 = min(width, x + w + margin_x)
        y1 = min(height, y + h_box + margin_y)

    keep_mask = np.zeros((height, width), dtype=np.uint8)
    keep_mask[y0:y1, x0:x1] = 255
    mask = cv2.bitwise_not(keep_mask)

    keep_ratio = float(np.count_nonzero(keep_mask)) / float(keep_mask.size)
    stats_payload = {
        "roi_bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
        "keep_ratio": keep_ratio,
        "mask_ratio": 1.0 - keep_ratio,
        "best_component_area": int(best_component_area),
        "used_fallback": used_fallback,
    }
    return mask, stats_payload


def _apply_black_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = image_bgr.copy()
    masked[mask > 0] = 0
    return masked


def _build_sfm_params(run_root: Path, imgdir: Path, variant: str, min_points3d: int = 50000) -> Path:
    payload = {
        "sfm_params": {
            "profile_name": run_root.name,
            "recommended_params": {
                "work": str((run_root / "SfM_models" / variant).resolve()),
                "validation_report": str(
                    (run_root / "reports" / "pointcloud_validation_report.json").resolve()
                ),
                "imgdir": str(imgdir.resolve()),
                "mapper_type": "incremental",
                "matcher": "sequential",
                "max_features": 8192,
                "seq_overlap": 10,
                "loop_detection": False,
                "sift_peak_threshold": 0.006666666666666667,
                "sift_edge_threshold": 10,
                "min_points3d": int(min_points3d),
            },
        }
    }
    path = run_root / "sfm_params.json"
    _write_json(path, payload)
    return path


def _build_train_params(run_root: Path, imgdir: Path, variant: str) -> Path:
    payload = {
        "train_params": {
            "profile_name": run_root.name,
            "recommended_params": {
                "imgdir": str(imgdir.resolve()),
                "colmap": str((run_root / "SfM_models" / variant / "sparse" / "best").resolve()),
                "outdir": str((run_root / "3DGS_models").resolve()),
                "validation_report": str(
                    (run_root / "reports" / "pointcloud_validation_report.json").resolve()
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
    }
    path = run_root / "train_params.json"
    _write_json(path, payload)
    return path


def _write_run_summary(
    run_root: Path,
    variant: str,
    image_count: int,
    mask_summary_path: Path,
    val_stats_path: Path | None,
) -> None:
    summary: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "route": "mask_route_a",
        "variant": variant,
        "image_count": image_count,
        "mask_summary": str(mask_summary_path.resolve()),
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
    _write_json(run_root / "run_summary.json", summary)


@app.command()
def main(
    variant: str = typer.Option("A1_highlight_mask", help="Mask Route A variant."),
    imgdir: str = typer.Option(str(DEFAULT_IMGDIR), help="Input image directory."),
    run_root: str = typer.Option("", help="Experiment root. Empty = auto timestamped."),
    skip_train: bool = typer.Option(False, help="Stop after sparse reconstruction."),
    max_images: int = typer.Option(0, help="For smoke/debug only. 0 = all images."),
    highlight_percentile: float = typer.Option(99.6, help="Adaptive brightness percentile."),
    brightness_floor: int = typer.Option(235, help="Minimum brightness threshold."),
    saturation_ceiling: int = typer.Option(120, help="Maximum saturation for highlight regions."),
    min_blob_area_ratio: float = typer.Option(0.0002, help="Minimum connected-component area ratio."),
    dilation_size: int = typer.Option(9, help="Dilation kernel size."),
    canny_low: int = typer.Option(50, help="A2/A3 machine ROI Canny low threshold."),
    canny_high: int = typer.Option(150, help="A2/A3 machine ROI Canny high threshold."),
    red_sat_floor: int = typer.Option(70, help="A2/A3 minimum saturation for red machine regions."),
    red_val_floor: int = typer.Option(45, help="A2/A3 minimum value for red machine regions."),
    min_component_area_ratio: float = typer.Option(
        0.003, help="A2/A3 minimum evidence component area ratio."
    ),
    roi_margin_ratio: float = typer.Option(0.05, help="A2/A3 ROI expansion margin ratio."),
    fallback_center_x: float = typer.Option(0.62, help="A2/A3 fallback ROI center x ratio."),
    fallback_width_ratio: float = typer.Option(0.70, help="A2/A3 fallback ROI width ratio."),
    fallback_height_ratio: float = typer.Option(0.94, help="A2/A3 fallback ROI height ratio."),
    min_points3d: int = typer.Option(50000, help="SfM Step 3 minimum points3D gate."),
) -> None:
    valid_variants = {"A1_highlight_mask", "A2_machine_roi", "A3_combined"}
    if variant not in valid_variants:
        raise SystemExit(f"Unsupported variant: {variant}. Expected one of: {sorted(valid_variants)}")

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
        run_root_path = DEFAULT_ROOT / f"{variant}_{_timestamp()}"

    masks_dir = run_root_path / "masks"
    masked_frames_dir = run_root_path / "masked_frames"
    reports_dir = run_root_path / "reports"
    logs_dir = run_root_path / "logs"
    for path in (masks_dir, masked_frames_dir, reports_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    images = _list_images(imgdir_path)
    if max_images > 0:
        images = images[:max_images]
    image_count = len(images)

    params = {
        "route": "mask_route_a",
        "variant": variant,
        "imgdir": str(imgdir_path.resolve()),
        "image_count": image_count,
        "skip_train": skip_train,
        "highlight_percentile": highlight_percentile,
        "brightness_floor": brightness_floor,
        "saturation_ceiling": saturation_ceiling,
        "min_blob_area_ratio": min_blob_area_ratio,
        "dilation_size": dilation_size,
        "canny_low": canny_low,
        "canny_high": canny_high,
        "red_sat_floor": red_sat_floor,
        "red_val_floor": red_val_floor,
        "min_component_area_ratio": min_component_area_ratio,
        "roi_margin_ratio": roi_margin_ratio,
        "fallback_center_x": fallback_center_x,
        "fallback_width_ratio": fallback_width_ratio,
        "fallback_height_ratio": fallback_height_ratio,
    }
    _write_json(run_root_path / "mask_route_params.json", params)

    print("=" * 72, flush=True)
    print(f"Mask Route A runner: {variant}", flush=True)
    print(f"Run root    : {run_root_path}", flush=True)
    print(f"Images      : {imgdir_path} ({image_count} files)", flush=True)
    print(f"Masks dir   : {masks_dir}", flush=True)
    print(f"Masked dir  : {masked_frames_dir}", flush=True)
    print("=" * 72, flush=True)

    mask_stats: list[dict[str, Any]] = []
    mask_log = logs_dir / "mask.log"
    with mask_log.open("w", encoding="utf-8", errors="replace", buffering=1) as log_file:
        for image_path in tqdm(images, desc=variant, ascii=True):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise SystemExit(f"Failed to read image: {image_path}")

            if variant == "A1_highlight_mask":
                mask, stats_payload = _build_highlight_mask(
                    image_bgr=image,
                    highlight_percentile=highlight_percentile,
                    brightness_floor=brightness_floor,
                    saturation_ceiling=saturation_ceiling,
                    min_blob_area_ratio=min_blob_area_ratio,
                    dilation_size=dilation_size,
                )
            elif variant == "A2_machine_roi":
                mask, stats_payload = _build_machine_roi_mask(
                    image_bgr=image,
                    canny_low=canny_low,
                    canny_high=canny_high,
                    red_sat_floor=red_sat_floor,
                    red_val_floor=red_val_floor,
                    min_component_area_ratio=min_component_area_ratio,
                    roi_margin_ratio=roi_margin_ratio,
                    fallback_center_x=fallback_center_x,
                    fallback_width_ratio=fallback_width_ratio,
                    fallback_height_ratio=fallback_height_ratio,
                )
            else:
                highlight_mask, highlight_stats = _build_highlight_mask(
                    image_bgr=image,
                    highlight_percentile=highlight_percentile,
                    brightness_floor=brightness_floor,
                    saturation_ceiling=saturation_ceiling,
                    min_blob_area_ratio=min_blob_area_ratio,
                    dilation_size=dilation_size,
                )
                roi_mask, roi_stats = _build_machine_roi_mask(
                    image_bgr=image,
                    canny_low=canny_low,
                    canny_high=canny_high,
                    red_sat_floor=red_sat_floor,
                    red_val_floor=red_val_floor,
                    min_component_area_ratio=min_component_area_ratio,
                    roi_margin_ratio=roi_margin_ratio,
                    fallback_center_x=fallback_center_x,
                    fallback_width_ratio=fallback_width_ratio,
                    fallback_height_ratio=fallback_height_ratio,
                )
                mask = cv2.bitwise_or(highlight_mask, roi_mask)
                stats_payload = {
                    **{f"highlight_{k}": v for k, v in highlight_stats.items()},
                    **{f"roi_{k}": v for k, v in roi_stats.items()},
                    "mask_ratio": float(np.count_nonzero(mask)) / float(mask.size),
                }
            masked = _apply_black_mask(image, mask)

            mask_path = masks_dir / f"{image_path.stem}.png"
            masked_path = masked_frames_dir / image_path.name
            cv2.imwrite(str(mask_path), mask)
            cv2.imwrite(str(masked_path), masked)

            row = {"image": image_path.name, **stats_payload}
            mask_stats.append(row)
            log_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            log_file.flush()

    ratios = [row["mask_ratio"] for row in mask_stats]
    summary = {
        "timestamp": datetime.now().isoformat(),
        "route": "mask_route_a",
        "variant": variant,
        "source_imgdir": str(imgdir_path.resolve()),
        "masked_imgdir": str(masked_frames_dir.resolve()),
        "image_count": image_count,
        "avg_mask_ratio": float(np.mean(ratios)) if ratios else 0.0,
        "min_mask_ratio": float(np.min(ratios)) if ratios else 0.0,
        "max_mask_ratio": float(np.max(ratios)) if ratios else 0.0,
        "mask_stats": mask_stats,
    }
    mask_summary_path = reports_dir / "mask_summary.json"
    _write_json(mask_summary_path, summary)
    print(f"[OK] Mask summary: {mask_summary_path}", flush=True)

    python = str(_resolve_python())
    sfm_params = _build_sfm_params(
        run_root_path,
        masked_frames_dir,
        variant,
        min_points3d=min_points3d,
    )
    sfm_cmd = [python, "-u", "-m", "src.sfm_colmap", "--params-json", str(sfm_params.resolve())]
    _run_process(sfm_cmd, logs_dir / "sfm.log")

    if skip_train:
        _write_run_summary(run_root_path, variant, image_count, mask_summary_path, None)
        print("[INFO] skip_train=True, stopping after sparse reconstruction.", flush=True)
        return

    train_params = _build_train_params(run_root_path, masked_frames_dir, variant)
    train_cmd = [python, "-u", "-m", "src.train_3dgs", "--params-json", str(train_params.resolve())]
    _run_process(train_cmd, logs_dir / "train.log")

    stats_path = _latest_val_stats(run_root_path / "3DGS_models" / "stats")
    _write_run_summary(run_root_path, variant, image_count, mask_summary_path, stats_path)
    print(f"[OK] Mask Route A run finished: {variant}", flush=True)


if __name__ == "__main__":
    app()
