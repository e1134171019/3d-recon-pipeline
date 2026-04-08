#!/usr/bin/env python3
"""
Phase 0: Video Preprocessing - Extract all valid frames based on quality criteria
Strategy: Extract at fps=1 (all frames) → Apply quality filters → frames_cleaned
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

def get_video_path():
    """Locate video file in data/viode directory"""
    data_dir = Path("data/viode")
    if not data_dir.exists():
        print(f"❌ 找不到視頻目錄: {data_dir}")
        return None
    
    video_files = list(data_dir.glob("*.mp4")) + list(data_dir.glob("*.avi"))
    if not video_files:
        print(f"❌ {data_dir} 中找不到視頻文件")
        return None
    
    if len(video_files) > 1:
        print(f"⚠️  找到多個視頻文件，使用: {video_files[0].name}")
    
    return video_files[0]

def apply_clahe(frame):
    """Apply Contrast Limited Adaptive Histogram Equalization"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def apply_gamma_correction(frame, gamma=0.5):
    """Apply gamma correction for underexposed images"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(frame, table)

def suppress_highlights(frame, threshold=240):
    """Suppress overly bright pixels in HSV V channel"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v[v > threshold] = threshold
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def get_frame_quality_metrics(frame):
    """Compute quality metrics for a frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Sharpness (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Brightness
    mean_brightness = np.mean(gray)
    
    # Contrast
    contrast = np.std(gray)
    
    return {
        "laplacian_var": laplacian_var,
        "mean_brightness": mean_brightness,
        "contrast": contrast
    }

def filter_frame_quality(frame, blur_threshold=40, brightness_low=30, brightness_high=220):
    """
    Filter frame based on quality criteria:
    - Sufficient sharpness (Laplacian variance > threshold)
    - Appropriate brightness (not too dark, not too bright)
    - Reasonable contrast
    """
    metrics = get_frame_quality_metrics(frame)
    
    # Check sharpness
    if metrics["laplacian_var"] < blur_threshold:
        return False, "blur"
    
    # Check brightness
    if metrics["mean_brightness"] < brightness_low or metrics["mean_brightness"] > brightness_high:
        return False, "brightness"
    
    # Aggressive filtering: accept frame if passes sharpness and brightness
    return True, "pass"

def sample_validation_set(cleaned_frames_dir, val_output_dir="data/frames_val", sample_ratio=0.17):
    """
    【Step 3】Sample validation frames uniformly from cleaned frames
    
    Args:
        cleaned_frames_dir: Directory containing cleaned frames
        val_output_dir: Output directory for validation frames
        sample_ratio: Sampling ratio (0.17 = 145/853)
    
    Returns:
        dict with validation sampling statistics
    """
    cleaned_path = Path(cleaned_frames_dir)
    val_path = Path(val_output_dir)
    val_path.mkdir(parents=True, exist_ok=True)
    
    # List all cleaned frames
    cleaned_frames = sorted(cleaned_path.glob("frame_*.jpg"))
    total_cleaned = len(cleaned_frames)
    
    print(f"\n{'='*70}")
    print(f"【Step 3 - 采样驗證集】")
    print(f"{'='*70}")
    print(f"📂 輸入: {cleaned_frames_dir} ({total_cleaned} 張幀)")
    print(f"📊 采样比例: {sample_ratio:.1%} (≈ {int(total_cleaned * sample_ratio)} 張)")
    print()
    
    # Calculate uniform sampling indices
    num_samples = max(50, int(total_cleaned * sample_ratio))  # At least 50 frames
    indices = np.linspace(0, total_cleaned - 1, num_samples, dtype=int)
    
    sampled_count = 0
    print(f"🔄 采样中...")
    for idx, frame_idx in enumerate(indices):
        src_frame = cleaned_frames[frame_idx]
        dst_frame = val_path / f"frame_{idx:06d}.jpg"
        
        # Copy frame
        img = cv2.imread(str(src_frame))
        if img is not None:
            cv2.imwrite(str(dst_frame), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            sampled_count += 1
    
    print(f"✅ 采样完成: {sampled_count} 張驗證幀")
    print(f"📁 驗證集保存到: {val_output_dir}")
    print()
    
    val_stats = {
        "timestamp": datetime.now().isoformat(),
        "cleaned_frames_dir": str(cleaned_frames_dir),
        "total_cleaned": total_cleaned,
        "sample_ratio": sample_ratio,
        "sampled_frames": sampled_count,
        "output_dir": str(val_output_dir)
    }
    
    return val_stats

def preprocess_phase0(video_path, output_dir="data/frames_cleaned", fps=1, 
                      gamma=0.5, blur_threshold=40):
    """
    【Step 1-2】Extract frames from video with quality filtering
    
    Args:
        video_path: Path to video file
        output_dir: Output directory for cleaned frames
        fps: Frames per second to extract (1 = all frames)
        gamma: Gamma correction value
        blur_threshold: Minimum Laplacian variance for frame acceptance
    
    Returns:
        dict with processing statistics
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"[Step 1] Extract All Frames")
    print(f"{'='*70}")
    print(f"Input Video: {video_path.name}")
    print(f"Parameters: fps={fps}")
    print()
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ 無法打開視頻: {video_path}")
        return None
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate frame extraction interval based on desired fps
    if fps > 0 and video_fps > 0:
        frame_interval = max(1, int(video_fps / fps))
    else:
        frame_interval = 1
    
    print(f"📊 視頻信息:")
    print(f"  總幀數: {total_frames}")
    print(f"  視頻 FPS: {video_fps}")
    print(f"  提取 FPS: {fps}")
    print(f"  幀間隔: {frame_interval}")
    print()
    
    # Step 1: Extract all frames (temporary, raw)
    print(f"📹 Step 1: 提取原始幀...")
    frame_count = 0
    extracted_count = 0
    temp_frames = []
    
    pbar = tqdm(total=total_frames, desc="  提取幀", unit="frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract based on frame interval
        if frame_count % frame_interval == 0:
            extracted_count += 1
            temp_frames.append(frame)
        
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    # Save Step 1 statistics
    extraction_stats = {
        "timestamp": datetime.now().isoformat(),
        "step": "extraction",
        "video_path": str(video_path),
        "total_frames_in_video": total_frames,
        "extracted_frames": extracted_count,
        "fps": fps,
        "frame_interval": frame_interval,
        "status": "PASS" if extracted_count >= 500 else "FAIL",
        "min_required_frames": 500
    }
    
    extraction_stats_file = output_path / "extraction_stats.json"
    with open(extraction_stats_file, "w") as f:
        json.dump(extraction_stats, f, indent=2)
    
    print(f"✅ Step 1 完成: 提取 {extracted_count} 張原始幀")
    print(f"  ✓ 検查: 幀數 ≥ 500 ? {extraction_stats['status']}")
    print(f"  📄 統計保存: extraction_stats.json")
    print()
    
    # Step 2: Apply quality filters
    print(f"⚙️  Step 2: 质量过滤 + 预处理...")
    accepted_count = 0
    rejected_reasons = {"blur": 0, "brightness": 0}
    
    pbar = tqdm(total=len(temp_frames), desc="  過濾幀", unit="frame")
    
    for frame in temp_frames:
        # Check quality
        is_valid, reason = filter_frame_quality(frame, blur_threshold)
        
        if is_valid:
            # Apply preprocessing
            frame = apply_gamma_correction(frame, gamma)
            frame = apply_clahe(frame)
            frame = suppress_highlights(frame, threshold=240)
            
            # Save frame
            output_file = output_path / f"frame_{accepted_count:06d}.jpg"
            cv2.imwrite(str(output_file), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            accepted_count += 1
        else:
            rejected_reasons[reason] += 1
        
        pbar.update(1)
    
    pbar.close()
    
    # Save Step 2 statistics
    acceptance_rate = 100.0 * accepted_count / extracted_count if extracted_count > 0 else 0
    filtering_stats = {
        "timestamp": datetime.now().isoformat(),
        "step": "filtering",
        "input_frames": extracted_count,
        "accepted_frames": accepted_count,
        "rejected_frames": extracted_count - accepted_count,
        "acceptance_rate": acceptance_rate,
        "rejected_reasons": rejected_reasons,
        "gamma": gamma,
        "blur_threshold": blur_threshold,
        "status": "PASS" if acceptance_rate >= 50.0 and accepted_count >= 100 else "FAIL",
        "min_required_acceptance_rate": 50.0,
        "min_required_frames": 100
    }
    
    filtering_stats_file = output_path / "filtering_stats.json"
    with open(filtering_stats_file, "w") as f:
        json.dump(filtering_stats, f, indent=2)
    
    print(f"✅ Step 2 完成: 保留 {accepted_count} 張有效幀")
    print(f"  ✓ 検查: 有效率 > 50% ? {filtering_stats['status']} ({acceptance_rate:.1f}%)")
    print(f"  ✓ 検查: 有效幀數 ≥ 100 ? {filtering_stats['status']}")
    if rejected_reasons["blur"] > 0:
        print(f"     - 模糊: {rejected_reasons['blur']}")
    if rejected_reasons["brightness"] > 0:
        print(f"     - 亮度: {rejected_reasons['brightness']}")
    print(f"  📄 統計保存: filtering_stats.json")
    print()
    
    stats = {
        "timestamp": datetime.now().isoformat(),
        "extraction_stats": extraction_stats,
        "filtering_stats": filtering_stats,
        "accepted_frames": accepted_count,
        "output_dir": output_dir
    }
    
    return stats

if __name__ == "__main__":
    # Find video
    video_path = get_video_path()
    if not video_path:
        sys.exit(1)
    
    # Step 1-2: Run preprocessing with fps=1 (extract all frames)
    stats = preprocess_phase0(
        video_path=video_path,
        output_dir="data/frames_cleaned",
        fps=1,
        gamma=0.5,
        blur_threshold=40
    )
    
    if not stats or stats["accepted_frames"] <= 0:
        print(f"❌ Step 1-2 失敗: 未提取到任何有效幀")
        sys.exit(1)
    
    print(f"✅ Step 1-2 成功: {stats['accepted_frames']} 張有效幀")
    
    # Step 3: Sample validation set
    val_stats = sample_validation_set(
        cleaned_frames_dir="data/frames_cleaned",
        val_output_dir="data/frames_val",
        sample_ratio=0.17
    )
    
    if val_stats["sampled_frames"] <= 0:
        print(f"❌ Step 3 失敗: 未采样到任何驗證幀")
        sys.exit(1)
    
    # Save Step 3 statistics
    val_path = Path("data/frames_val")
    val_stats["status"] = "PASS" if val_stats["sampled_frames"] >= 50 else "FAIL"
    val_stats_file = val_path / "validation_stats.json"
    with open(val_stats_file, "w") as f:
        json.dump(val_stats, f, indent=2)
    
    print(f"✅ Step 3 完成: {val_stats['sampled_frames']} 張驗證幀")
    print(f"  ✓ 検查: 驗證集 ≥ 50 ? {val_stats['status']}")
    print(f"  📄 統計保存: validation_stats.json")
    print()
    
    # Phase 0 Summary
    print(f"{'='*70}")
    print(f"【Phase 0 - 完成總結】")
    print(f"{'='*70}")
    print(f"✅ Step 1: 提取 {stats['extraction_stats']['extracted_frames']} 張原始幀")
    print(f"✅ Step 2: 過濾保留 {stats['accepted_frames']} 張有效幀 ({stats['filtering_stats']['acceptance_rate']:.1f}%)")
    print(f"✅ Step 3: 采样 {val_stats['sampled_frames']} 張驗證幀")
    print(f"📁 输出:")
    print(f"   ├─ data/frames_cleaned/ ({stats['accepted_frames']} 張)")
    print(f"   ├─ data/frames_val/ ({val_stats['sampled_frames']} 張)")
    print(f"   └─ 統計文件:")
    print(f"      ├─ extraction_stats.json (Step 1)")
    print(f"      ├─ filtering_stats.json (Step 2)")
    print(f"      └─ validation_stats.json (Step 3)")
    print(f"{'='*70}")
