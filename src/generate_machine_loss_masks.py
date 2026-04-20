from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import typer

from src.run_mask_route_opencv import _build_machine_roi_mask

app = typer.Typer(help="Generate machine-level background masks for 3DGS loss masking.")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _list_images(imgdir: Path) -> list[Path]:
    return [
        path
        for path in sorted(imgdir.iterdir())
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]


@app.command()
def main(
    imgdir: str = typer.Option(..., help="Input image directory."),
    outdir: str = typer.Option(..., help="Output PNG mask directory."),
) -> None:
    imgdir_path = Path(imgdir)
    if not imgdir_path.is_absolute():
        imgdir_path = (PROJECT_ROOT / imgdir_path).resolve()
    outdir_path = Path(outdir)
    if not outdir_path.is_absolute():
        outdir_path = (PROJECT_ROOT / outdir_path).resolve()

    if not imgdir_path.exists():
        raise SystemExit(f"Image directory not found: {imgdir_path}")

    images = _list_images(imgdir_path)
    if not images:
        raise SystemExit(f"No images found under: {imgdir_path}")

    outdir_path.mkdir(parents=True, exist_ok=True)

    mask_rows: list[dict[str, Any]] = []
    excluded_ratios: list[float] = []
    keep_ratios: list[float] = []
    fallback_count = 0

    for image_path in images:
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise SystemExit(f"Failed to read image: {image_path}")

        mask, stats = _build_machine_roi_mask(
            image_bgr=frame,
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

        mask_path = outdir_path / f"{image_path.stem}.png"
        cv2.imwrite(str(mask_path), mask)

        excluded_ratio = float(np.count_nonzero(mask)) / float(mask.size)
        keep_ratio = 1.0 - excluded_ratio
        excluded_ratios.append(excluded_ratio)
        keep_ratios.append(keep_ratio)
        if stats.get("used_fallback", False):
            fallback_count += 1

        mask_rows.append(
            {
                "image": image_path.name,
                "mask_path": str(mask_path),
                "excluded_ratio": excluded_ratio,
                "keep_ratio": keep_ratio,
                "roi_bbox": stats.get("roi_bbox"),
                "used_fallback": bool(stats.get("used_fallback", False)),
                "best_component_area": int(stats.get("best_component_area", 0)),
            }
        )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "imgdir": str(imgdir_path),
        "outdir": str(outdir_path),
        "image_count": len(images),
        "avg_excluded_ratio": float(np.mean(excluded_ratios)),
        "min_excluded_ratio": float(np.min(excluded_ratios)),
        "max_excluded_ratio": float(np.max(excluded_ratios)),
        "avg_keep_ratio": float(np.mean(keep_ratios)),
        "fallback_count": int(fallback_count),
        "rows": mask_rows,
    }
    _write_json(outdir_path.parent / "loss_mask_summary.json", summary)
    print(f"[OK] Generated {len(images)} machine loss masks -> {outdir_path}", flush=True)
    print(
        f"[INFO] avg_keep_ratio={summary['avg_keep_ratio']:.4f}, "
        f"avg_excluded_ratio={summary['avg_excluded_ratio']:.4f}, "
        f"fallback_count={fallback_count}",
        flush=True,
    )


if __name__ == "__main__":
    app()
