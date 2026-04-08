#!/usr/bin/env python3
"""
Create validation set: Uniformly sample 145 frames from frames_cleaned for rapid testing
"""

import os
import sys
import shutil
import json
from pathlib import Path
from datetime import datetime
import numpy as np

def create_validation_set(cleaned_dir="data/frames_cleaned", val_size=145, 
                          output_dir="data/frames_val"):
    """
    Create validation set by uniform sampling from cleaned frames
    
    Args:
        cleaned_dir: Directory containing all cleaned frames
        val_size: Number of frames for validation set (minimum, default 145)
        output_dir: Output directory for validation frames
    
    Returns:
        dict with validation statistics
    """
    
    cleaned_path = Path(cleaned_dir)
    output_path = Path(output_dir)
    
    # Check if cleaned frames exist
    if not cleaned_path.exists():
        print(f"❌ 找不到清理幀目錄: {cleaned_path}")
        return None
    
    # Get list of frames
    frame_files = sorted([f for f in cleaned_path.glob("frame_*.jpg")])
    if not frame_files:
        print(f"❌ {cleaned_path} 中找不到任何幀")
        return None
    
    total_cleaned = len(frame_files)
    
    print(f"\n{'='*70}")
    print(f"【創建驗證集】")
    print(f"{'='*70}")
    print(f"📂 清理幀目錄: {cleaned_dir}")
    print(f"📊 總清理幀數: {total_cleaned}")
    print(f"🎯 驗證集大小: {val_size}")
    print()
    
    # Uniform sampling
    if total_cleaned <= val_size:
        print(f"⚠️  清理幀數 ({total_cleaned}) ≤ 驗證集大小 ({val_size})")
        print(f"   使用所有 {total_cleaned} 張幀作為驗證集")
        val_indices = np.arange(total_cleaned)
    else:
        # Uniform sampling
        val_indices = np.linspace(0, total_cleaned - 1, val_size, dtype=int)
    
    selected_frames = [frame_files[i] for i in val_indices]
    
    print(f"📋 採樣策略: 均勻採樣")
    print(f"  範圍: 0 到 {total_cleaned - 1}")
    print(f"  採樣點: {len(selected_frames)} 張")
    print()
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Copy frames
    print(f"📋 複製驗證幀...")
    for i, src_frame in enumerate(selected_frames):
        dst_frame = output_path / f"frame_{i:06d}.jpg"
        shutil.copy2(src_frame, dst_frame)
    
    print(f"✅ 複製完成: {len(selected_frames)} 張幀")
    print()
    
    # Save statistics
    stats = {
        "timestamp": datetime.now().isoformat(),
        "cleaned_dir": cleaned_dir,
        "total_cleaned_frames": total_cleaned,
        "validation_size": len(selected_frames),
        "selected_indices": val_indices.tolist(),
        "output_dir": output_dir,
        "sampling_strategy": "uniform"
    }
    
    stats_file = output_path / "stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"📊 【驗證集統計】")
    print(f"  清理幀數: {total_cleaned}")
    print(f"  驗證幀數: {len(selected_frames)}")
    print(f"  採樣率: {len(selected_frames) / total_cleaned * 100:.1f}%")
    print(f"  輸出目錄: {output_dir}")
    print()
    
    print(f"✅ 驗證集創建完成！")
    print(f"📝 下一步: 運行驗證工作流")
    print(f"   Phase 1A-Test (COLMAP): python src/run_sfm_validation.py")
    print(f"   Phase 1B-Test (3DGS):   python src/train_3dgs_validation.py")
    print()
    
    return stats

if __name__ == "__main__":
    # Create validation set from cleaned frames
    stats = create_validation_set(
        cleaned_dir="data/frames_cleaned",
        val_size=145,
        output_dir="data/frames_val"
    )
    
    if not stats:
        sys.exit(1)
