"""
Phase 0 - 視頻診斷工具
診斷視頻是否能正常讀取，檢查曝光質量
"""

import cv2
import numpy as np
from pathlib import Path


def diagnose_video(video_path, sample_frames=20):
    """
    診斷視頻質量
    
    檢查項目:
    1. 視頻是否能正常讀取
    2. 曝光質量 (均勻性、亮度)
    3. 幀數和分辨率
    
    Args:
        video_path (str): 視頻路徑
        sample_frames (int): 採樣幀數 (均勻分佈)
    
    Returns:
        dict: 診斷結果
    """
    print(f"\n【視頻診斷】{Path(video_path).name}")
    print("=" * 60)
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ 【失敗】無法打開視頻文件")
            return {"status": "failed", "reason": "cannot_open"}
        
        # 獲取視頻基本信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps if fps > 0 else 0
        
        print(f"\n📹 【基本信息】")
        print(f"  總幀數: {total_frames} 幀")
        print(f"  FPS: {fps:.1f}")
        print(f"  分辨率: {width} × {height}")
        print(f"  時長: {duration_sec:.1f} 秒")
        
        # 計算採樣步長
        step = max(1, total_frames // sample_frames)
        
        # 採樣幀並分析曝光
        results = []
        failed_frames = []
        
        for i in range(sample_frames):
            frame_idx = i * step
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                failed_frames.append(frame_idx)
                continue
            
            # 轉灰度並分析
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 計算統計量
            mean = gray.mean()
            std = gray.std()
            pure_white = (gray == 255).sum() / gray.size
            pure_black = (gray == 0).sum() / gray.size
            
            results.append({
                'frame_idx': frame_idx,
                'mean': mean,
                'std': std,
                'pure_white': pure_white,
                'pure_black': pure_black
            })
        
        cap.release()
        
        # 統計分析
        if not results:
            print("❌ 【失敗】無法讀取任何幀")
            return {"status": "failed", "reason": "no_frames_readable"}
        
        means = np.array([r['mean'] for r in results])
        stds = np.array([r['std'] for r in results])
        whites = np.array([r['pure_white'] for r in results])
        
        avg_mean = means.mean()
        avg_std = stds.mean()
        avg_white = whites.mean()
        
        print(f"\n📊 【曝光分析】(採樣 {len(results)} 幀)")
        print(f"  平均亮度: {avg_mean:.1f} / 255")
        print(f"  標準差 (對比度): {avg_std:.1f}")
        print(f"  純白比例: {avg_white*100:.2f}%")
        
        if failed_frames:
            print(f"\n⚠️  無法讀取的幀: {len(failed_frames)} 幀")
        
        # 診斷結果
        print(f"\n🔍 【診斷結果】")
        
        issues = []
        
        # 檢查曝光
        if avg_white > 0.5:
            print("  ❌ 過曝光: 超過50%純白像素 → 資訊丟失，建議 gamma > 0.6")
            issues.append("overexposed")
        elif avg_white > 0.2:
            print("  ⚠️  輕微過曝: 20-50%純白 → 建議 gamma = 0.5-0.6")
            issues.append("slightly_overexposed")
        elif avg_mean > 220:
            print("  ⚠️  偏亮: 平均亮度 > 220 → 建議 gamma = 0.5-0.6")
            issues.append("bright")
        elif avg_mean < 30:
            print("  ❌ 過低曝: 平均亮度 < 30 → 信噪比差，難以重建")
            issues.append("underexposed")
        elif avg_mean < 80:
            print("  ⚠️  偏暗: 平均亮度 < 80 → 建議曝光補償")
            issues.append("dark")
        else:
            print(f"  ✅ 曝光正常: 平均亮度 {avg_mean:.1f}")
        
        # 檢查對比度
        if avg_std < 20:
            print("  ⚠️  對比度低: 標準差 < 20 → 缺乏細節，COLMAP 特徵提取困難")
            issues.append("low_contrast")
        else:
            print(f"  ✅ 對比度良好: 標準差 {avg_std:.1f}")
        
        # 總結
        if not issues:
            status = "excellent"
            recommendation = "✅ 視頻質量優秀，可直接使用 (gamma = 0.6, CLAHE 標準設置)"
        elif all(issue in ["bright", "slightly_overexposed"] for issue in issues):
            status = "good_with_correction"
            recommendation = "⚠️  亮度偏高，建議使用 gamma = 0.6 和 CLAHE 進行曝光修正"
        elif "low_contrast" in issues:
            status = "warning"
            recommendation = "⚠️  對比度低，可能影響 SfM 特徵提取，仍可嘗試"
        elif "underexposed" in issues:
            status = "failed"
            recommendation = "❌ 曝光不足，不建議使用此視頻"
        else:
            status = "warning"
            recommendation = "⚠️  存在多個問題，建議檢查視頻內容"
        
        print(f"\n📋 【建議】")
        print(f"  {recommendation}")
        
        # 計算 Phase 0 參數
        print(f"\n⚙️  【Phase 0 建議參數】")
        print(f"  fps: 2 (每秒 2 幀)")
        print(f"  預期提取幀數: {int(total_frames / fps * 2)}  # (基於 2fps)")
        
        if avg_mean > 200:
            print(f"  gamma: 0.7  # 曝光偏高")
        elif avg_mean > 150:
            print(f"  gamma: 0.6  # 標準設置")
        elif avg_mean > 100:
            print(f"  gamma: 0.5  # 曝光偏低")
        else:
            print(f"  gamma: 0.4  # 需要大幅亮度提升")
        
        print(f"  CLAHE: clipLimit=2.0, tileGridSize=(8,8)  # 標準設置")
        print(f"  suppress_highlight: threshold=240  # 壓低過亮區域")
        
        print("=" * 60)
        
        return {
            "status": status,
            "total_frames": total_frames,
            "fps": fps,
            "duration_sec": duration_sec,
            "predicted_frames_at_2fps": int(total_frames / fps * 2),
            "avg_brightness": avg_mean,
            "avg_contrast": avg_std,
            "pure_white_ratio": avg_white,
            "issues": issues,
            "recommendation": recommendation
        }
    
    except Exception as e:
        print(f"❌ 【錯誤】{str(e)}")
        return {"status": "error", "reason": str(e)}


if __name__ == "__main__":
    import sys
    
    # 查找 data/viode 目錄中的視頻
    video_dir = Path(__file__).parent.parent / "data" / "viode"
    
    if not video_dir.exists():
        print(f"❌ 視頻目錄不存在: {video_dir}")
        sys.exit(1)
    
    # 尋找視頻文件 (支持多種格式)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f"*{ext}"))
        video_files.extend(video_dir.glob(f"*{ext.upper()}"))
    
    if not video_files:
        print(f"❌ 在 {video_dir} 中找不到任何視頻文件")
        sys.exit(1)
    
    print(f"\n🎬 找到 {len(video_files)} 個視頻文件")
    print(f"位置: {video_dir}\n")
    
    # 診斷每個視頻
    results = []
    for video_file in sorted(video_files):
        result = diagnose_video(str(video_file), sample_frames=20)
        results.append((video_file.name, result))
    
    # 總結
    print(f"\n\n{'='*60}")
    print("【診斷完成 - 總結】")
    print(f"{'='*60}")
    
    for filename, result in results:
        status_icon = {
            "excellent": "✅",
            "good_with_correction": "⚠️",
            "warning": "⚠️",
            "failed": "❌",
            "error": "❌"
        }.get(result.get("status"), "❓")
        
        print(f"\n{status_icon} {filename}")
        if "total_frames" in result:
            print(f"    總幀數: {result['total_frames']}")
            print(f"    預計提取: {result['predicted_frames_at_2fps']} 幀 (@ 2fps)")
            print(f"    平均亮度: {result['avg_brightness']:.1f}/255")
