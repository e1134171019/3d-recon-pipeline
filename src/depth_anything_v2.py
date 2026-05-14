# src/depth_anything_v2.py
# -*- coding: utf-8 -*-
"""
Depth Anything V2 integration guard.

This module is intentionally not wired into the current formal mainline.
The local repository does not ship a working Depth Anything V2 runtime, so
attempting to use this path should fail fast with a clear error instead of
silently degrading into a no-op.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


DEPTH_ANYTHING_V2_UNAVAILABLE = (
    "Depth Anything V2 is not available in the current formal mainline. "
    "This module is a guard only. If you want depth supervision, implement "
    "and validate the full dependency/runtime path first."
)


class DepthAnythingV2Estimator:
    """Fail-fast guard for an unavailable Depth Anything V2 backend."""

    def __init__(self, model_name: str = "vit_large", device: str = "cuda:0"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._availability_error = DEPTH_ANYTHING_V2_UNAVAILABLE
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Keep the unavailable state explicit."""
        self.model = None

    def _lazy_load_model(self) -> None:
        """Depth Anything V2 is intentionally unavailable in this repo."""
        raise NotImplementedError(self._availability_error)

    def estimate_depth(
        self,
        image_path: Path,
        save_depth_map: Optional[Path] = None,
    ) -> np.ndarray:
        """
        Guard entrypoint.

        Validates the image path so callers get a concrete file error first,
        then fails fast because the backend is intentionally unavailable.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        self._lazy_load_model()
        raise RuntimeError(
            "Unreachable guard: Depth Anything V2 backend was expected to raise."
        )

    def estimate_batch_depth(
        self,
        image_dir: Path,
        output_dir: Optional[Path] = None,
        frame_pattern: str = "*.jpg",
    ) -> dict[str, np.ndarray]:
        """Guard batch entrypoint."""
        image_paths = sorted(image_dir.glob(frame_pattern))
        if not image_paths:
            raise FileNotFoundError(
                f"No images found in {image_dir} matching {frame_pattern}"
            )

        self._lazy_load_model()
        raise RuntimeError(
            "Unreachable guard: Depth Anything V2 backend was expected to raise."
        )


def compute_depth_consistency_loss(
    predicted_depth: np.ndarray,
    estimated_depth: np.ndarray,
    mask: Optional[np.ndarray] = None,
    lambda_weight: float = 0.1,
) -> float:
    """
    Compute depth consistency loss between rendered depth and estimated depth.

    This helper remains valid because it only operates on arrays. The unavailable
    part is the local depth-estimation backend, not the loss math itself.
    """
    if predicted_depth.shape != estimated_depth.shape:
        raise ValueError(
            f"Shape mismatch: predicted {predicted_depth.shape} vs estimated {estimated_depth.shape}"
        )

    pred_norm = predicted_depth / (np.max(predicted_depth) + 1e-8)
    est_norm = estimated_depth / (np.max(estimated_depth) + 1e-8)
    diff = np.abs(pred_norm - est_norm)

    if mask is not None:
        diff = diff * mask.astype(np.float32)
        loss = np.mean(diff[mask > 0])
    else:
        loss = np.mean(diff)

    return lambda_weight * loss


__all__ = [
    "DEPTH_ANYTHING_V2_UNAVAILABLE",
    "DepthAnythingV2Estimator",
    "compute_depth_consistency_loss",
]
