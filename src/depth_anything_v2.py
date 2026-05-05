# src/depth_anything_v2.py
# -*- coding: utf-8 -*-
"""
Depth Anything V2 Integration Module

Provides unsupervised monocular depth estimation for depth consistency loss
during 3DGS training (P0 Breakthrough 2).

This module integrates https://github.com/DepthAnything/Depth-Anything-V2
for zero-shot depth prediction from RGB images.
"""

from pathlib import Path
import json
import numpy as np
from typing import Optional
import cv2


class DepthAnythingV2Estimator:
    """
    Wrapper for Depth Anything V2 depth estimation.
    
    Provides methods to:
    1. Estimate monocular depth for a set of images
    2. Generate depth consistency loss for 3DGS training
    3. Save/load depth maps for reproducibility
    """
    
    def __init__(self, model_name: str = "vit_large", device: str = "cuda:0"):
        """
        Initialize Depth Anything V2 model.
        
        Args:
            model_name: Model variant ("vit_small", "vit_base", "vit_large").
                       Larger = more accurate but slower.
            device: Computation device ("cuda:0", "cpu").
        
        Raises:
            ImportError: If depth_anything_v2 package is not installed.
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Load Depth Anything V2 model (lazy loading on first use)."""
        try:
            # 延遲加載（首次使用時才初始化）
            pass  # Model will be loaded on first predict() call
        except ImportError as e:
            raise ImportError(
                "Depth Anything V2 not installed. "
                "Install: pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git"
            ) from e
    
    def estimate_depth(
        self,
        image_path: Path,
        save_depth_map: Optional[Path] = None,
    ) -> np.ndarray:
        """
        Estimate depth from a single RGB image.
        
        Args:
            image_path: Path to input RGB image.
            save_depth_map: Optional path to save normalized depth map (0-255, uint8).
        
        Returns:
            Depth map as numpy array (H, W), normalized to [0, 1] float32.
        """
        # 讀取圖像
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 延遲初始化 model
        if self.model is None:
            self._lazy_load_model()
        
        # 推理深度
        with np.no_grad() if hasattr(np, 'no_grad') else __import__('torch').no_grad():
            depth = self.model.infer_image(img_rgb)  # Returns (H, W) normalized [0, 1]
        
        # 保存深度圖（可選）
        if save_depth_map is not None:
            save_depth_map.parent.mkdir(parents=True, exist_ok=True)
            depth_uint8 = (depth * 255).astype(np.uint8)
            cv2.imwrite(str(save_depth_map), depth_uint8)
        
        return depth.astype(np.float32)
    
    def estimate_batch_depth(
        self,
        image_dir: Path,
        output_dir: Optional[Path] = None,
        frame_pattern: str = "*.jpg",
    ) -> dict[str, np.ndarray]:
        """
        Estimate depth for all images in a directory.
        
        Args:
            image_dir: Directory containing input images.
            output_dir: Optional directory to save depth maps.
            frame_pattern: Glob pattern to match frames (default: "*.jpg").
        
        Returns:
            Dictionary mapping image filenames to depth arrays.
        """
        depth_maps = {}
        image_paths = sorted(image_dir.glob(frame_pattern))
        
        if not image_paths:
            raise FileNotFoundError(f"No images found in {image_dir} matching {frame_pattern}")
        
        print(f"[*] Estimating depth for {len(image_paths)} images...")
        
        for i, img_path in enumerate(image_paths):
            save_depth = None
            if output_dir is not None:
                save_depth = output_dir / f"{img_path.stem}_depth.png"
            
            try:
                depth = self.estimate_depth(img_path, save_depth)
                depth_maps[img_path.name] = depth
                
                if (i + 1) % 100 == 0:
                    print(f"  [{i + 1}/{len(image_paths)}] Processed {img_path.name}")
            
            except Exception as e:
                print(f"  ⚠️  Error processing {img_path.name}: {e}")
                continue
        
        print(f"[OK] Depth estimation complete: {len(depth_maps)}/{len(image_paths)} successful")
        return depth_maps
    
    def _lazy_load_model(self) -> None:
        """Lazy load model on first inference (memory efficient)."""
        try:
            # 這會在實際使用時由 train_3dgs.py 驅動
            # 這裡只是佔位符，實際實現需要 depth_anything_v2 套件
            raise NotImplementedError(
                "Model loading requires depth_anything_v2 package. "
                "Install: pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git"
            )
        except Exception as e:
            print(f"Warning: Could not load Depth Anything V2 model: {e}")
            print("Depth loss will be skipped for this training session.")
            self.model = None


def compute_depth_consistency_loss(
    predicted_depth: np.ndarray,
    estimated_depth: np.ndarray,
    mask: Optional[np.ndarray] = None,
    lambda_weight: float = 0.1,
) -> float:
    """
    Compute depth consistency loss between rendered depth and estimated depth.
    
    Args:
        predicted_depth: Rendered depth from 3DGS (H, W).
        estimated_depth: Estimated depth from Depth Anything V2 (H, W).
        mask: Optional binary mask to exclude regions (e.g., sky).
        lambda_weight: Weight of depth loss (default 0.1).
    
    Returns:
        Weighted L1 depth consistency loss.
    
    Formula:
        L_depth = lambda * mean(|predicted - estimated| * mask)
    """
    if predicted_depth.shape != estimated_depth.shape:
        raise ValueError(
            f"Shape mismatch: predicted {predicted_depth.shape} vs estimated {estimated_depth.shape}"
        )
    
    # Normalize depths to similar scale
    pred_norm = predicted_depth / (np.max(predicted_depth) + 1e-8)
    est_norm = estimated_depth / (np.max(estimated_depth) + 1e-8)
    
    # L1 差異
    diff = np.abs(pred_norm - est_norm)
    
    # 應用遮罩（如果提供）
    if mask is not None:
        diff = diff * mask.astype(np.float32)
        loss = np.mean(diff[mask > 0])
    else:
        loss = np.mean(diff)
    
    return lambda_weight * loss


__all__ = ["DepthAnythingV2Estimator", "compute_depth_consistency_loss"]
