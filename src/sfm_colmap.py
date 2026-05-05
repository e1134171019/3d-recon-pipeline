# sfm_colmap.py
# -*- coding: utf-8 -*-
from src.utils import ensure_utf8_stdout

ensure_utf8_stdout()

"""
Phase 1A - SfM 重建（唯一入口）
=============================
使用 COLMAP 從清潔幀重建稀疏點雲。

快速啟動：
  python -m src.sfm_colmap

指定 COLMAP：
  python -m src.sfm_colmap --colmap-bin colmap/bin/colmap.exe

斷點續跑（已有 database 時跳過特徵提取和匹配）：
  python -m src.sfm_colmap --resume

Phase 1A → Phase 1B 過渡條件（自動檢查）
======================================
Phase 1A 完成後必須滿足以下條件才能進入 Phase 1B（train_3dgs.py）：

1. **特徵提取驗證** (check_features):
   ✓ Average features per image ≥ 500
   ✓ Total descriptors in DB ≥ 10,000
   ✓ DB size ≥ 50 MB (indicates dense feature database)

2. **特徵匹配驗證** (check_matching):
   ✓ Matched image pairs ≥ 3
   ✓ Inlier ratio ≥ 0.1 (COLMAP official threshold from GitHub)
   ✓ Average inliers per match ≥ 10

3. **稀疏重建驗證** (check_reconstruction):
   ✓ Cameras ≥ 3 (minimum for 3D reconstruction)
   ✓ Registered images ≥ 3
   ✓ 3D points ≥ min_points3d (default: 50,000)
   ✓ can_proceed_to_3dgs = true

If all checks pass:
  → PointCloud validation report written to outputs/reports/
  → train_3dgs.py will read this report to authorize Phase 1B
  → Scene scale automatically detected or taken from scale_json

If any check fails:
  → Phase 1A terminates with detailed diagnostic
  → Suggestions for adjusting parameters (--max_features, --max_image_size, etc.)
"""
from pathlib import Path
from dataclasses import dataclass, replace
import subprocess
import shutil
import json
from datetime import datetime
import time
import sqlite3
import typer
from src.utils.agent_contracts import write_stage_contract

app = typer.Typer(help="Phase 1A: COLMAP SfM 重建")

DEFAULT_SFM_PARAMS = {
    "imgdir": "data/frames_1600",
    "work": "outputs/SfM_models/sift",
    "use_gpu": True,
    "max_image_size": 1600,
    "max_features": 8192,
    "seq_overlap": 10,
    "loop_detection": True,
    "mapper_type": "incremental",
    "sift_peak_threshold": 0.006666666666666667,
    "sift_edge_threshold": 10,
    "colmap_bin": None,
    "glomap_bin": None,
    "resume": False,
    "min_points3d": 50000,
}

_SFM_PARAM_TYPES = {
    **dict.fromkeys(("imgdir", "work", "mapper_type", "colmap_bin", "glomap_bin"), str),
    **dict.fromkeys(("max_image_size", "max_features", "seq_overlap", "sift_edge_threshold", "min_points3d"), int),
    **dict.fromkeys(("use_gpu", "loop_detection", "resume"), bool),
    **dict.fromkeys(("sift_peak_threshold",), float),
}


@dataclass(frozen=True)
class SfmConfig:
    """Phase 1A COLMAP SfM reconstruction configuration.
    
    This config drives feature extraction, matching, and sparse reconstruction.
    Validation gates ensure quality before Phase 1B (3DGS training).
    
    Attributes:
        imgdir: Input image directory path.
        work: Working directory for COLMAP intermediate files.
        use_gpu: Enable GPU acceleration for feature extraction/matching.
        max_image_size: Resize images to this size if larger (0=no resize). 
        max_features: SIFT features per image (~2000-4000 typical).
        seq_overlap: Image sequence overlap for incremental mapper (10-15 typical).
        loop_detection: Enable loop closure detection (slower but more robust).
        mapper_type: "incremental" or "glomap" (glomap for global optimization).
        sift_peak_threshold: SIFT peak threshold (default 0.0067, lower=more features).
        sift_edge_threshold: SIFT edge threshold (default 10, higher=stricter).
        colmap_bin: Path to colmap executable (auto-discovered if None).
        glomap_bin: Path to glomap executable (required if mapper_type="glomap").
        resume: Resume from existing DB if present (skip feature extraction).
        min_points3d: Minimum 3D points for Phase 1A→1B gate (default 50,000).
            Typical: 50,000-200,000 depending on scene complexity.
        params_json: Path to agent-provided param overrides (empty=defaults).
    
    Critical gates (checked before proceeding to Phase 1B):
        - avg_features_per_image ≥ 500 (checked in check_features())
        - matched_image_pairs_inlier_ratio ≥ 0.1 (COLMAP official threshold)
        - num_points3d ≥ min_points3d (checked in check_reconstruction())
    """
    imgdir: str
    work: str
    use_gpu: bool
    max_image_size: int
    max_features: int
    seq_overlap: int
    loop_detection: bool
    mapper_type: str
    sift_peak_threshold: float
    sift_edge_threshold: int
    colmap_bin: str | None
    glomap_bin: str | None
    resume: bool
    min_points3d: int
    params_json: str = ""


@dataclass(frozen=True)
class SfmPaths:
    project_root: Path
    img: Path
    work_p: Path
    outputs_root: Path
    sfm_models_dir: Path
    reports_dir: Path
    db: str
    vocab_tree: Path | None

def run(cmd: list[str]) -> None:
    """Run a command and echo it; raise if non-zero."""
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def find_colmap(colmap_bin: str | None) -> str:
    """Resolve path to colmap executable (from arg, local, or PATH)."""
    if colmap_bin:
        if Path(colmap_bin).exists():
            return colmap_bin
        raise SystemExit(f"指定的 COLMAP 不存在：{colmap_bin}")
    # 先找專案內的 colmap
    local = Path("colmap") / "bin" / "colmap.exe"
    if local.exists():
        return str(local.resolve())
    # 再從 PATH 找
    found = shutil.which("colmap")
    if not found:
        raise SystemExit(
            "找不到 COLMAP：請 ① 將 colmap.exe 所在資料夾加入 PATH，或 ② 使用 --colmap-bin 指定完整路徑。\n"
            "例如：--colmap-bin \"C:/tools/colmap/bin/colmap.exe\""
        )
    return found


def find_glomap(glomap_bin: str | None) -> str:
    """Resolve path to glomap executable (from arg, installers, or PATH)."""
    if glomap_bin:
        if Path(glomap_bin).exists():
            return glomap_bin
        raise SystemExit(f"指定的 GLOMAP 不存在：{glomap_bin}")

    local_candidates = [
        Path("installers") / "glomap-1.2.0" / "bin" / "glomap.exe",
        Path("glomap") / "bin" / "glomap.exe",
    ]
    for candidate in local_candidates:
        if candidate.exists():
            return str(candidate.resolve())

    found = shutil.which("glomap")
    if found:
        return found

    raise SystemExit(
        "找不到 GLOMAP：請先安裝官方 Windows binary，或使用 --glomap-bin 指定 glomap.exe。\n"
        "例如：--glomap-bin \"C:/3d-recon-pipeline/installers/glomap-1.2.0/bin/glomap.exe\""
    )


def _load_sfm_params(params_json: str) -> tuple[dict, dict]:
    """Load agent-provided SfM params.

    Supports either:
      1. a sfm_params.json file whose top-level already is the plan
      2. a larger payload that contains a "sfm_params" section
    """
    params_path = Path(params_json)
    if not params_path.exists():
        raise SystemExit(f"找不到 params json：{params_path}")

    payload = json.loads(params_path.read_text(encoding="utf-8"))
    plan = payload.get("sfm_params", payload)
    recommended = plan.get("recommended_params", {})
    if not isinstance(recommended, dict):
        raise SystemExit(f"params json 格式不正確：{params_path}")
    return plan, recommended


def _apply_recommended_sfm_params(config: SfmConfig, recommended: dict) -> SfmConfig:
    """Apply agent-recommended params by casting and merging with replace()."""
    updates: dict[str, object] = {}
    for key, default_value in DEFAULT_SFM_PARAMS.items():
        current_value = getattr(config, key)
        if current_value == default_value and key in recommended:
            caster = _SFM_PARAM_TYPES.get(key, str)
            updates[key] = caster(recommended[key])
    if not updates:
        return config
    return replace(config, **updates)


def _resolve_sfm_config(
    *,
    imgdir: str,
    work: str,
    use_gpu: bool,
    max_image_size: int,
    max_features: int,
    seq_overlap: int,
    loop_detection: bool,
    mapper_type: str,
    sift_peak_threshold: float,
    sift_edge_threshold: int,
    colmap_bin: str | None,
    glomap_bin: str | None,
    resume: bool,
    min_points3d: int,
    params_json: str,
) -> SfmConfig:
    config = SfmConfig(
        imgdir=imgdir,
        work=work,
        use_gpu=use_gpu,
        max_image_size=max_image_size,
        max_features=max_features,
        seq_overlap=seq_overlap,
        loop_detection=loop_detection,
        mapper_type=mapper_type,
        sift_peak_threshold=sift_peak_threshold,
        sift_edge_threshold=sift_edge_threshold,
        colmap_bin=colmap_bin,
        glomap_bin=glomap_bin,
        resume=resume,
        min_points3d=min_points3d,
        params_json=params_json,
    )
    if not params_json:
        return config

    plan, recommended = _load_sfm_params(params_json)
    print(
        f"[AGENT] 套用 SfM 設定: {Path(params_json).resolve()} "
        f"(profile={plan.get('profile_name', 'unknown')})"
    )
    return _apply_recommended_sfm_params(config, recommended)


def _resolve_sfm_paths(config: SfmConfig) -> SfmPaths:
    project_root = Path(__file__).parent.parent
    img = project_root / config.imgdir if not Path(config.imgdir).is_absolute() else Path(config.imgdir)
    work_p = project_root / config.work if not Path(config.work).is_absolute() else Path(config.work)

    if not img.exists() or not any(img.iterdir()):
        raise SystemExit(f"影像夾為空：{img.resolve()}，請先抽影格/降尺寸")

    (work_p / "sparse").mkdir(parents=True, exist_ok=True)
    outputs_root = _infer_outputs_root(project_root, work_p)
    outputs_root.mkdir(parents=True, exist_ok=True)
    sfm_models_dir = outputs_root / "SfM_models"
    reports_dir = outputs_root / "reports"
    sfm_models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    vocab_tree = project_root / "colmap" / "vocab_tree_flickr100K_words32K.bin"
    if not vocab_tree.exists():
        print("[WARN] 找不到 vocab_tree 文件，將停用 loop detection")
        print(f"       預期位置：{vocab_tree.resolve()}")
        print("       下載：https://demuc.de/colmap/")
        vocab_tree = None

    return SfmPaths(
        project_root=project_root,
        img=img,
        work_p=work_p,
        outputs_root=outputs_root,
        sfm_models_dir=sfm_models_dir,
        reports_dir=reports_dir,
        db=str(work_p / "database.db"),
        vocab_tree=vocab_tree,
    )


def _build_feature_extractor_args(config: SfmConfig, paths: SfmPaths, colmap_exe: str) -> list[str]:
    return [
        colmap_exe, "feature_extractor",
        "--database_path", paths.db,
        "--image_path", str(paths.img),
        "--ImageReader.single_camera", "1",
        "--FeatureExtraction.use_gpu", "1" if config.use_gpu else "0",
        "--SiftExtraction.max_image_size", str(config.max_image_size),
        "--SiftExtraction.max_num_features", str(config.max_features),
        "--SiftExtraction.peak_threshold", str(config.sift_peak_threshold),
        "--SiftExtraction.edge_threshold", str(config.sift_edge_threshold),
    ]


def _build_sequential_matcher_args(config: SfmConfig, paths: SfmPaths, colmap_exe: str) -> list[str]:
    args = [
        colmap_exe, "sequential_matcher",
        "--database_path", paths.db,
        "--SequentialMatching.overlap", str(config.seq_overlap),
        "--FeatureMatching.use_gpu", "1" if config.use_gpu else "0",
    ]
    if config.loop_detection and paths.vocab_tree is not None:
        args += [
            "--SequentialMatching.loop_detection", "1",
            "--SequentialMatching.vocab_tree_path", str(paths.vocab_tree),
        ]
    return args


def _write_sfm_complete_contract(
    *,
    paths: SfmPaths,
    config: SfmConfig,
    mapper_type: str,
    best_sparse: Path,
    result3: dict,
    dense_ply: Path | None = None,
) -> dict[str, str]:
    artifacts = {
        "pointcloud_report": paths.reports_dir / "pointcloud_validation_report.json",
        "sparse_dir": best_sparse,
        "database": Path(paths.db),
    }
    
    # 添加密集點雲到 artifacts（P0 突破口）
    if dense_ply is not None and dense_ply.exists():
        artifacts["dense_ply"] = dense_ply
    
    metrics = {
        "cameras_count": result3["cameras_count"],
        "images_count": result3["images_count"],
        "registered_images_count": result3["registered_images_count"],
        "points3d_count": result3["points3d_count"],
        "can_proceed_to_3dgs": result3["can_proceed_to_3dgs"],
    }
    
    # 添加密集點數統計
    if dense_ply is not None and dense_ply.exists():
        file_size_mb = dense_ply.stat().st_size / (1024 * 1024)
        metrics["dense_ply_size_mb"] = file_size_mb
    
    return write_stage_contract(
        project_root=paths.project_root,
        run_root=paths.outputs_root,
        stage="sfm_complete",
        status="completed" if result3["can_proceed_to_3dgs"] else "blocked",
        artifacts=artifacts,
        metrics=metrics,
        params={
            "imgdir": str(paths.img.resolve()),
            "work": str(paths.work_p.resolve()),
            "mapper_type": mapper_type,
            "use_gpu": config.use_gpu,
            "max_image_size": config.max_image_size,
            "max_features": config.max_features,
            "seq_overlap": config.seq_overlap,
            "loop_detection": config.loop_detection,
        },
        summary=(
            "SfM reconstruction complete and ready for 3DGS"
            if result3["can_proceed_to_3dgs"]
            else "SfM reconstruction complete but blocked by gate"
        ),
    )


def _infer_outputs_root(project_root: Path, work_p: Path) -> Path:
    """Infer the outputs root from the selected SfM work directory.

    Examples:
      outputs/SfM_models/sift                  -> outputs
      outputs/runs/run_x/SfM_models/sift       -> outputs/runs/run_x
    """
    for candidate in [work_p] + list(work_p.parents):
        if candidate.name == "SfM_models":
            return candidate.parent
    return project_root / "outputs"


def _read_sparse_model_stats(sparse_path: Path) -> tuple[dict, list[str]]:
    """Read sparse model statistics from a specific COLMAP reconstruction directory."""
    warnings: list[str] = []
    stats = {
        "cameras_count": 0,
        "images_count": 0,
        "registered_images_count": 0,
        "points3d_count": 0,
    }

    try:
        import pycolmap

        reconstruction = pycolmap.Reconstruction(str(sparse_path))
        stats["cameras_count"] = int(getattr(reconstruction, "num_cameras", lambda: len(reconstruction.cameras))())
        if hasattr(reconstruction, "num_images"):
            stats["images_count"] = int(reconstruction.num_images())
        else:
            stats["images_count"] = len(reconstruction.images)
        if hasattr(reconstruction, "num_reg_images"):
            stats["registered_images_count"] = int(reconstruction.num_reg_images())
        else:
            stats["registered_images_count"] = stats["images_count"]
        stats["points3d_count"] = int(getattr(reconstruction, "num_points3D", lambda: len(reconstruction.points3D))())
        return stats, warnings
    except Exception as exc:
        warnings.append(f"pycolmap read failed, falling back to file-size estimation: {exc}")

    cameras_size = (sparse_path / "cameras.bin").stat().st_size if (sparse_path / "cameras.bin").exists() else 0
    images_size = (sparse_path / "images.bin").stat().st_size if (sparse_path / "images.bin").exists() else 0
    points3d_size = (sparse_path / "points3D.bin").stat().st_size if (sparse_path / "points3D.bin").exists() else 0

    # File-size fallback is only for rough ranking when pycolmap is unavailable.
    stats["cameras_count"] = cameras_size // 64 if cameras_size > 0 else 0
    stats["images_count"] = images_size // 190000 if images_size > 0 else 0
    stats["registered_images_count"] = stats["images_count"]
    stats["points3d_count"] = points3d_size // 148 if points3d_size > 0 else 0
    return stats, warnings


def _find_best_sparse_model(sparse_root: Path) -> Path | None:
    """Pick the best reconstruction among sparse/{0,1,2,...} outputs."""
    if not sparse_root.exists():
        return None

    candidates = []
    for child in sorted(sparse_root.iterdir()):
        if not child.is_dir():
            continue
        direct_required = [child / "cameras.bin", child / "images.bin", child / "points3D.bin"]
        if all(path.exists() and path.stat().st_size > 0 for path in direct_required):
            stats, _ = _read_sparse_model_stats(child)
            candidates.append((child, stats))
            continue

        # GLOMAP Windows binary may create output_path/0/{cameras,images,points3D}.bin.
        for nested in sorted(child.iterdir()):
            if not nested.is_dir():
                continue
            nested_required = [nested / "cameras.bin", nested / "images.bin", nested / "points3D.bin"]
            if not all(path.exists() and path.stat().st_size > 0 for path in nested_required):
                continue
            stats, _ = _read_sparse_model_stats(nested)
            candidates.append((nested, stats))

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item[1]["registered_images_count"],
            item[1]["points3d_count"],
            item[0].name,
        ),
        reverse=True,
    )
    return candidates[0][0]

# 共用报告分隔符（避免重复定义）
_SEP = "\n" + "="*72

def _report(label: str, stats: dict, errors: list, warnings: list | None = None) -> dict:
    """
    共用的驗證報告生成工具
    
    Args:
        label: 步驟名稱（如 "Step 1 特徵提取驗證"）
        stats: 統計數據字典（如 {"num_matches": 1000, ...}）
        errors: 錯誤列表
        warnings: 警告列表（可選）
    
    Returns:
        統一格式的驗證結果字典
    """
    if warnings is None:
        warnings = []
    
    pass_check = len(errors) == 0
    
    if errors:
        msg = f"{_SEP}\n[FAIL] {label}\n"
        for err in errors:
            msg += f"  • {err}\n"
        msg += f"{_SEP}\n"
    else:
        msg = f"{_SEP}\n[OK] {label}\n"
        if warnings:
            for w in warnings:
                msg += f"  ⚠️  {w}\n"
        msg += f"{_SEP}\n"
    
    print(msg, flush=True)
    
    return {
        "pass": pass_check,
        "errors": errors,
        "warnings": warnings,
        **stats
    }

def _get_matcher_time_estimate(use_gpu: bool) -> str:
    """
    返回 Sequential Matcher 的保守時間估計（非精確預測）
    
    注意：實際時間取決於硬件、場景複雜度等多因素
    此估計僅供參考，不應用於嚴格的時間規劃
    
    Args:
        use_gpu: 是否使用 GPU
    
    Returns:
        人類可讀的時間估計字符串
    """
    # Sequential matcher 只比對相鄰幀，比 vocab_tree 快得多
    if use_gpu:
        return "5-15 分鐘（GPU 加速，sequential matcher）"
    else:
        return "15-30 分鐘（CPU 模式，sequential matcher）"

def _assess_mapper_feasibility(num_images: int, inlier_ratio: float, use_gpu: bool) -> str:
    """
    評估 Mapper 是否可行及參數健度
    
    不返回時間預測（不可靠），而是返回可行性評估
    
    Args:
        num_images: 圖像數量
        inlier_ratio: 內點率
        use_gpu: 是否使用 GPU
    
    Returns:
        可行性評估字符串
    """
    msg = "參數評估:\n"
    
    # 幀數評估
    if num_images < 30:
        msg += f"  • 圖像數 ({num_images}): ⚠️  較少，可能無法完成建圖\n"
    elif num_images > 1000:
        msg += f"  • 圖像數 ({num_images}): ⚠️  較多，建圖可能耗時很長\n"
    else:
        msg += f"  • 圖像數 ({num_images}): ✓ 合理範圍\n"
    
    # 內點率評估
    if inlier_ratio < 0.05:
        msg += f"  • 內點率 ({inlier_ratio:.3f}): ❌ 極低，Mapper 可能失敗\n"
    elif inlier_ratio < 0.10:
        msg += f"  • 內點率 ({inlier_ratio:.3f}): ⚠️  偏低，建圖風險高\n"
    elif inlier_ratio < 0.30:
        msg += f"  • 內點率 ({inlier_ratio:.3f}): ✓ 勉強可接受\n"
    else:
        msg += f"  • 內點率 ({inlier_ratio:.3f}): ✓ 優質\n"
    
    # GPU/CPU 評估
    msg += f"  • 硬件: {'GPU 加速' if use_gpu else 'CPU 模式（較慢）'}\n"
    msg += "\n⏱️  注意：時間預測不可靠，實際執行因多因素變化而異\n"
    
    return msg

def _query_db(db: str, queries: dict[str, str]) -> dict[str, int]:
    """Execute multiple SQL queries in one transaction.
    
    Args:
        db: database.db path
        queries: dict mapping keys to SQL queries
        
    Returns:
        dict mapping keys to query results (first column as int, defaulting to 0 if NULL)
    """
    with sqlite3.connect(db) as conn:
        return {k: (conn.execute(sql).fetchone() or [0])[0] for k, sql in queries.items()}


def check_features(db: str) -> dict:
    """
    驗證特徵提取是否成功（Step 1 檢查）
    
    基於 COLMAP GitHub 官方推薦：
    https://github.com/colmap/colmap/blob/dev/src/colmap/feature/extraction.cc
    
    檢查項目:
      ✅ database.db ≥ 10MB (升級自 100KB)
      ✅ keypoints 表有內容
      ✅ descriptors 表有內容
      ✅ avg_features_per_image > 500 (新增關鍵⭐)
      ✅ 數據一致性檢查
    
    Args:
        db: database.db 路徑
    
    Returns:
        {
            "pass": bool,
            "cameras_count": int,
            "total_features": int,
            "avg_features_per_image": float,
            "db_size_mb": float,
            "errors": list[str]
        }
    """
    
    db_path = Path(db)
    
    # 1. 檢查 database.db 是否存在
    if not db_path.exists():
        return _report(
            "Step 1 特徵提取驗證",
            {"cameras_count": 0, "total_features": 0, "avg_features_per_image": 0, "db_size_mb": 0},
            ["database.db 不存在"]
        )
    
    db_size_mb = db_path.stat().st_size / 1024 / 1024
    if db_size_mb < 10:  # ← 升級自 100KB → 10MB
        return _report(
            "Step 1 特徵提取驗證",
            {"cameras_count": 0, "total_features": 0, "avg_features_per_image": 0, "db_size_mb": db_size_mb},
            [f"database.db 過小: {db_size_mb:.1f}MB < 10MB"]
        )
    
    # 2. 連接數據庫並查詢統計
    try:
        row = _query_db(db, {
            "images_count": "SELECT COUNT(*) FROM keypoints",
            "total_keypoints": "SELECT SUM(rows) FROM keypoints",
            "descriptors_rows": "SELECT COUNT(*) FROM descriptors",
        })
        images_count = row["images_count"]
        total_keypoints = row["total_keypoints"]
        descriptors_rows = row["descriptors_rows"]
    except Exception as e:
        return _report(
            "Step 1 特徵提取驗證",
            {"cameras_count": 0, "total_features": 0, "avg_features_per_image": 0, "db_size_mb": db_size_mb},
            [f"無法查詢 database: {str(e)}"]
        )
    
    # 3. 驗證數據一致性與品質
    errors = []
    
    if images_count == 0:
        errors.append("keypoints 表為空（未提取特徵）")
    
    if total_keypoints is None or total_keypoints == 0:
        errors.append("keypoints 總數為 0（未提取特徵）")
    
    if descriptors_rows == 0:
        errors.append("descriptors 表為空（缺少描述符）")
    
    if images_count != descriptors_rows:
        errors.append(f"數據不一致: keypoints ({images_count}) != descriptors ({descriptors_rows})")
    
    # 4. 計算平均特徵數 (新增關鍵檢查⭐)
    avg_features = (total_keypoints or 0) / images_count if images_count > 0 else 0
    
    if avg_features < 500:  # ← 官方推薦最小值
        errors.append(f"平均特徵數過少: {avg_features:.0f} < 500 (建議檢查圖像品質或調整 --max_features)")
    
    # 5. 總結結果（使用共用報告函數）
    return _report(
        "Step 1 特徵提取驗證",
        {
            "cameras_count": images_count,
            "total_features": total_keypoints or 0,
            "avg_features_per_image": avg_features,
            "db_size_mb": db_size_mb,
        },
        errors
    )


def check_matching(db: str) -> dict:
    """
    驗證匹配是否成功（Step 2 檢查）
    
    基於 COLMAP GitHub 官方推薦：
    https://github.com/colmap/colmap/blob/dev/src/colmap/feature/matching.cc
    
    檢查項目:
      ✅ database.db ≥ 20MB (升級自 10MB)
      ✅ num_matches ≥ 100
      ✅ num_inlier_matches > 0
      ✅ inlier_ratio > 0.1 (新增關鍵⭐⭐⭐ GitHub官方推薦)
      ✅ avg_matches_per_pair > 50
    
    Args:
        db: database.db 路徑（vocab_tree_matcher 完成後）
    
    Returns:
        {
            "pass": bool,
            "db_size_mb": float,
            "num_matches": int,
            "num_inlier_matches": int,
            "inlier_ratio": float,
            "avg_matches_per_pair": float,
            "errors": list[str],
            "warnings": list[str]
        }
    """
    
    db_path = Path(db)
    
    # 1. 檢查 database.db 是否存在
    try:
        db_size_mb = db_path.stat().st_size / 1024 / 1024
    except Exception:
        return _report(
            "Step 2 匹配驗證",
            {"db_size_mb": 0, "num_matches": 0, "num_inlier_matches": 0, "inlier_ratio": 0, "avg_matches_per_pair": 0},
            ["database.db 不存在或無法讀取"],
            []
        )
    
    if db_size_mb < 20:  # ← 升級自 10MB → 20MB
        return _report(
            "Step 2 匹配驗證",
            {"db_size_mb": db_size_mb, "num_matches": 0, "num_inlier_matches": 0, "inlier_ratio": 0, "avg_matches_per_pair": 0},
            [f"database.db 大小不足: {db_size_mb:.1f}MB < 20MB"],
            []
        )
    
    # 2. 連接數據庫查詢匹配統計
    try:
        row = _query_db(db, {
            "num_matches": "SELECT SUM(rows) FROM matches",
            "num_inlier": "SELECT SUM(rows) FROM two_view_geometries",
            "num_pairs": "SELECT COUNT(*) FROM matches WHERE rows > 0",
        })
        num_matches = row["num_matches"] or 0
        num_inlier_matches = row["num_inlier"] or 0
        num_attempted_pairs = row["num_pairs"] or 1
    except Exception as e:
        return _report(
            "Step 2 匹配驗證",
            {"db_size_mb": db_size_mb, "num_matches": 0, "num_inlier_matches": 0, "inlier_ratio": 0, "avg_matches_per_pair": 0},
            [f"無法查詢匹配表: {str(e)}"],
            []
        )
    
    # 3. 計算關鍵指標
    inlier_ratio = num_inlier_matches / num_matches if num_matches > 0 else 0
    avg_matches_per_pair = num_matches / num_attempted_pairs if num_attempted_pairs > 0 else 0
    
    # 4. 驗證品質
    errors = []
    warnings = []
    
    if num_matches < 100:
        errors.append(f"總匹配數不足: {num_matches} < 100")
    
    if num_inlier_matches == 0:
        errors.append("內點匹配數為 0")
    
    # 關鍵判定: 內點率分級（GitHub 官方推薦 > 0.1 🔴）
    if inlier_ratio < 0.05:
        errors.append(
            f"內點率極低: {inlier_ratio:.3f} < 0.05 (特徵品質差，Mapper 很可能失敗)"
        )
    elif inlier_ratio < 0.1:
        warnings.append(
            f"內點率偏低: {inlier_ratio:.3f} < 0.1 (建議檢查圖像品質或增加特徵數)"
        )
    
    if avg_matches_per_pair < 50:
        errors.append(
            f"平均匹配數過少: {avg_matches_per_pair:.1f} < 50 (相似幀對太少或視角重複)"
        )
    
    # 5. 總結結果（使用共用報告函數）
    return _report(
        "Step 2 匹配驗證",
        {
            "db_size_mb": db_size_mb,
            "num_matches": num_matches,
            "num_inlier_matches": num_inlier_matches,
            "inlier_ratio": inlier_ratio,
            "avg_matches_per_pair": avg_matches_per_pair,
        },
        errors,
        warnings
    )


def check_reconstruction(sparse_model_dir: str, min_points3d: int = 50000) -> dict:
    """
    驗證重建是否成功（Step 3 檢查）
    
    基於 COLMAP GitHub 官方推薦：
    https://github.com/colmap/colmap/blob/dev/src/colmap/sfm/incremental_mapper.cc
    
    檢查項目:
      ✅ cameras.bin / images.bin / points3D.bin 存在且有內容
      ✅ num_cameras ≥ 3 (最小相機數)
      ✅ num_points3d ≥ 50,000 (最小點雲)
    
    Args:
        sparse_model_dir: sparse/0/ 目錄路徑
    
    Returns:
        {
            "pass": bool,
            "cameras_count": int,
            "images_count": int,
            "points3d_count": int,
            "can_proceed_to_3dgs": bool,
            "errors": list[str],
            "warnings": list[str]
        }
    """
    
    sparse_path = Path(sparse_model_dir)
    errors = []
    warnings = []
    
    # 1. 檢查三個必要檔案是否存在且有大小
    required_files = {
        "cameras.bin": 32,        # 最少 32 bytes（单相机模式可能很小）
        "images.bin": 1024,       # 最少 1KB
        "points3D.bin": 10240,    # 最少 10KB
    }
    
    for filename, min_size in required_files.items():
        filepath = sparse_path / filename
        if not filepath.exists():
            errors.append(f"缺失檔案: {filename}")
        else:
            actual_size = filepath.stat().st_size
            if actual_size < min_size:
                errors.append(f"{filename} 太小: {actual_size} bytes < {min_size} bytes")
    
    # 2. 讀取實際重建統計（優先使用 pycolmap）
    stats, stats_warnings = _read_sparse_model_stats(sparse_path)
    warnings.extend(stats_warnings)
    
    # 3. 驗證重建品質
    min_registered_images = 3   # 實際可用的是註冊影像數，不是相機內參模型數
    
    if stats["registered_images_count"] < min_registered_images:
        errors.append(
            f"註冊影像數太少: {stats['registered_images_count']} < {min_registered_images} (匹配品質不足)"
        )
    
    if stats["points3d_count"] < min_points3d:
        errors.append(
            f"點雲太稀疏: {stats['points3d_count']} < {min_points3d}"
        )
    
    # 警告（非致命）
    if stats["points3d_count"] < 100000:
        warnings.append(
            f"點雲較少 ({stats['points3d_count']}), "
            "建議檢查特徵密度或視角覆蓋"
        )
    
    # 4. 決策
    can_proceed = len(errors) == 0
    
    # 5. 總結結果（使用共用報告函數）
    return _report(
        "Step 3 重建驗證",
        {
            "cameras_count": stats["cameras_count"],
            "images_count": stats["images_count"],
            "registered_images_count": stats["registered_images_count"],
            "points3d_count": stats["points3d_count"],
            "can_proceed_to_3dgs": can_proceed,
        },
        errors,
        warnings
    )

def export_signals(result3: dict, sparse_model_dir: str, reports_dir: Path) -> None:
    """
    根據 Step 3 驗證結果導出點雲驗證報告。
    
    供 train_3dgs.py 讀取，決定是否繼續進行 3DGS 訓練。
    
    Args:
        result3: check_reconstruction() 返回的驗證結果字典
        sparse_model_dir: sparse/0 的路徑字符串
        reports_dir: 報告輸出目錄
    """
    # 導出驗證報告（train_3dgs.py 會讀這個）
    report_file = reports_dir / "pointcloud_validation_report.json"
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "sparse_dir": sparse_model_dir,
        "cameras_count": result3["cameras_count"],
        "images_count": result3["images_count"],
        "registered_images_count": result3.get("registered_images_count", result3["images_count"]),
        "points3d_count": result3["points3d_count"],
        "can_proceed_to_3dgs": result3["can_proceed_to_3dgs"],
        "diagnosis": "SfM 重建成功，可進行 3DGS 訓練" if result3["can_proceed_to_3dgs"] else "SfM 重建失敗",
    }
    report_file.write_text(json.dumps(report_data, indent=2, ensure_ascii=False))
    print(f"[OK] 驗證報告已導出：{report_file.resolve()}")


def _run_stereo_fusion_step(
    colmap_exe: str,
    db: str,
    img: Path,
    work_p: Path,
    enable_fusion: bool = True,
) -> Path | None:
    """
    Execute COLMAP stereo_fusion to generate dense point cloud.
    
    This creates a million-scale dense initialization for 3DGS by fusing
    depth maps estimated from feature matching geometry.
    
    Args:
        colmap_exe: Path to colmap executable.
        db: Path to COLMAP database.
        img: Path to image directory.
        work_p: Working directory path.
        enable_fusion: Whether to run fusion (skip if False).
    
    Returns:
        Path to fused.ply if successful, None if skipped or failed.
    """
    if not enable_fusion:
        print("[*] Dense cloud fusion skipped (enable_fusion=False)")
        return None
    
    dense_root = work_p / "dense"
    dense_root.mkdir(parents=True, exist_ok=True)
    
    try:
        print("\n" + "─"*70)
        print("⏳ 執行 patch_match_stereo 深度估計...")
        print("─"*70)
        
        # Step 1: patch_match_stereo (估計每張圖的深度)
        run([
            colmap_exe, "patch_match_stereo",
            "--workspace_path", str(dense_root),
            "--workspace_format", "COLMAP",
            "--max_image_size", "2000",
            "--gpu_index", "0",
        ])
        
        print("\n" + "─"*70)
        print("⏳ 執行 stereo_fusion 生成密集點雲...")
        print("─"*70)
        
        # Step 2: stereo_fusion (融合深度圖成點雲)
        fused_ply = dense_root / "fused.ply"
        run([
            colmap_exe, "stereo_fusion",
            "--workspace_path", str(dense_root),
            "--workspace_format", "COLMAP",
            "--output_type", "PLY",
            "--output_path", str(fused_ply),
        ])
        
        if fused_ply.exists():
            file_size_mb = fused_ply.stat().st_size / (1024 * 1024)
            print(f"\n✅ Dense point cloud generated: {fused_ply.resolve()}")
            print(f"   File size: {file_size_mb:.1f} MB")
            return fused_ply
        else:
            print(f"⚠️  stereo_fusion 完成但找不到輸出檔案：{fused_ply}")
            return None
    
    except subprocess.CalledProcessError as e:
        print(f"\n⚠️  stereo_fusion 步驟失敗 (exit code {e.returncode})")
        print(f"   這通常不是致命錯誤 — 稀疏點雲可能仍然可用")
        print(f"   錯誤詳情：{e}")
        return None


def _run_mapper_step(
    mapper_type: str,
    colmap_exe: str,
    glomap_exe: str | None,
    db: str,
    img: Path,
    work_p: Path,
) -> None:
    """Run either COLMAP incremental mapper or standalone GLOMAP."""
    mapper_type_norm = mapper_type.strip().lower()
    sparse_root = work_p / "sparse"

    if mapper_type_norm == "incremental":
        run([
            colmap_exe, "mapper",
            "--database_path", db,
            "--image_path", str(img),
            "--output_path", str(sparse_root),
            "--Mapper.ba_local_max_num_iterations", "15",
            "--Mapper.ba_global_max_num_iterations", "30",
        ])
        return

    if mapper_type_norm in {"glomap", "global"}:
        if not glomap_exe:
            raise SystemExit("mapper_type=glomap 但未提供可用的 glomap.exe")
        sparse_root.mkdir(parents=True, exist_ok=True)
        run([
            glomap_exe, "mapper",
            "--database_path", db,
            "--image_path", str(img),
            "--output_path", str(sparse_root),
        ])
        return

    raise SystemExit(f"不支援的 mapper_type：{mapper_type}")

@app.command()
def main(
    imgdir: str = "data/frames_1600",
    work: str = "outputs/SfM_models/sift",  # ✅ SIFT 基準路徑
    use_gpu: bool = True,         # 改為 True（已安裝 GPU 版 COLMAP）
    max_image_size: int = 1600,   # 合理平衡（避免特徵過多導致匹配過慢）
    max_features: int = 8192,     # COLMAP 預設值（影片幀足夠）
    seq_overlap: int = 10,        # sequential_matcher 前後比對幀數
    loop_detection: bool = True,  # 若 vocab_tree 存在才可啟用
    mapper_type: str = "incremental",  # incremental | glomap
    sift_peak_threshold: float = 0.006666666666666667,
    sift_edge_threshold: int = 10,
    colmap_bin: str | None = None,
    glomap_bin: str | None = None,
    resume: bool = False,         # 斷點續跑：跳過已完成的步驟
    min_points3d: int = 50000,    # Step 3 驗證門檻；subset gate 可降低
    params_json: str = "",        # 由 agent 產生的 sfm_params.json
):
    """
    以 COLMAP 跑：
      1) feature_extractor
      2) sequential_matcher（影片相鄰幀匹配 + loop detection）
      3) mapper
    """
    config = _resolve_sfm_config(
        imgdir=imgdir,
        work=work,
        use_gpu=use_gpu,
        max_image_size=max_image_size,
        max_features=max_features,
        seq_overlap=seq_overlap,
        loop_detection=loop_detection,
        mapper_type=mapper_type,
        sift_peak_threshold=sift_peak_threshold,
        sift_edge_threshold=sift_edge_threshold,
        colmap_bin=colmap_bin,
        glomap_bin=glomap_bin,
        resume=resume,
        min_points3d=min_points3d,
        params_json=params_json,
    )
    paths = _resolve_sfm_paths(config)
    colmap_exe = find_colmap(config.colmap_bin)
    mapper_type = config.mapper_type.strip().lower()
    glomap_exe = find_glomap(config.glomap_bin) if mapper_type in {"glomap", "global"} else None

    try:
        # ⏱️ 開始時間戳
        start_time = time.time()
        
        # 1) 特徵提取
        if config.resume and Path(paths.db).exists() and Path(paths.db).stat().st_size > 1024:
            print("[RESUME] database.db 已存在，跳過特徵提取")
        else:
            run(_build_feature_extractor_args(config, paths, colmap_exe))
        
        # ✓ Step 1 檢查：特徵提取驗證（resume 時也要驗證）
        result1 = check_features(paths.db)
        if not result1["pass"]:
            raise SystemExit(
                "Step 1 特徵提取驗證失敗。請檢查：\n"
                "  1) 圖像品質是否足夠（不模糊、光線充足）\n"
                "  2) 調整 Phase 0 參數（gamma 校正、blur 閾值）\n"
                "  3) 調整 --max_features 參數（16000 → 24000）"
            )
        
        # Step 2 時間提示（保守估計，非精確預測）
        matcher_estimate = _get_matcher_time_estimate(use_gpu=config.use_gpu)
        print("接下來：Step 2 匹配驗證（Sequential Matcher + 相鄰幀匹配）\n")
        print(f"估計時間：{matcher_estimate}\n")

        # 2) Sequential 匹配（影片幀專用：只比對相鄰幀 + loop detection 閉環檢測）
        if not (
            config.resume
            and Path(paths.db).exists()
            and Path(paths.db).stat().st_size > 20 * 1024 * 1024
        ):
            # 只在未 resume 或 db 未包含匹配時才執行
            run(_build_sequential_matcher_args(config, paths, colmap_exe))
        
        # ✓ Step 2 檢查：匹配品質驗證（內點率關鍵⭐⭐⭐，resume 時也要驗證）
        result2 = check_matching(paths.db)
        if not result2["pass"]:
            raise SystemExit(
                "Step 2 匹配驗證失敗。問題分析：\n"
                f"  • 內點率: {result2['inlier_ratio']:.3f} (閾值: > 0.1)\n"
                "  建議調整：\n"
                "  1) 檢查 Phase 0 圖像預處理（光照、模糊）\n"
                "  2) 增加特徵數量（--max_features 16000 → 24000)\n"
                "  3) 嘗試降低 Mapper 閾值或增加重疊視角"
            )
        
        # Step 3 可行性評估（而非時間預測）
        mapper_feasibility = _assess_mapper_feasibility(
            result1['cameras_count'],
            result2['inlier_ratio'],
            config.use_gpu
        )
        mapper_label = "GLOMAP 全局式建圖" if mapper_type in {"glomap", "global"} else "增量式建圖"
        print(f"接下來：Step 3 {mapper_label}（Mapper + 稀疏點雲重建）")
        print(mapper_feasibility)

        # 3) 建圖
        _run_mapper_step(
            mapper_type=mapper_type,
            colmap_exe=colmap_exe,
            glomap_exe=glomap_exe,
            db=paths.db,
            img=paths.img,
            work_p=paths.work_p,
        )
        
        print("\n" + "─"*70)
        print("⏳ Mapper 完成 - 正在驗證重建結果...")
        print("─"*70 + "\n")
        
        # 4) 密集點雲重建 (Dense Cloud stereo_fusion) - P0 突破口 1
        print("[*] Step 3.5 - 執行密集點雲融合 (Dense Cloud stereo_fusion)...")
        dense_ply = _run_stereo_fusion_step(
            colmap_exe=colmap_exe,
            db=paths.db,
            img=paths.img,
            work_p=paths.work_p,
            enable_fusion=True,  # 可配置
        )
        
        # 🔍 診斷：Mapper 是否真的成功生成了 sparse/0
        sparse_root = paths.work_p / "sparse"
        best_sparse = _find_best_sparse_model(sparse_root)
        if best_sparse is None:
            raise SystemExit(
                f"❌ COLMAP Mapper 失敗診斷\n\n"
                f"預期輸出目錄不存在：{sparse_root.resolve()}\n\n"
                f"檢查清單：\n"
                f"  1) 檢查 {paths.work_p / 'sparse'} 是否存在\n"
                f"  2) 運行：ls {paths.work_p / 'sparse'}/\n"
                f"  3) 如果為空，Mapper 初始化失敗（匹配品質或參數問題）\n"
                f"  4) 嘗試調整：\n"
                f"     - 增加特徵數量：--max_features 24000\n"
                f"     - 降低解析度：--max_image_size 2000\n"
                f"     - 確保有足夠重疊角度\n"
                f"  5) 若使用 GLOMAP，請檢查 glomap.exe 與 output_path 是否正常"
            )
        
        # ✓ Step 3 檢查：重建驗證
        if best_sparse.name != "0":
            print(f"[INFO] 自動選擇最佳 sparse model：{best_sparse.resolve()}（不是 sparse/0）")
        result3 = check_reconstruction(str(best_sparse), min_points3d=config.min_points3d)
        if not result3["pass"]:
            raise SystemExit(
                "Step 3 重建驗證失敗。可能原因：\n" +
                "\n".join(f"  • {err}" for err in result3["errors"])
            )
        
        # ── 導出驗證報告供決策層使用 ──────────────────────
        print("\n[*] 正在導出點雲驗證報告...")
        export_signals(result3, str(best_sparse.resolve()), paths.reports_dir)
        contract_paths = _write_sfm_complete_contract(
            paths=paths,
            config=config,
            mapper_type=mapper_type,
            best_sparse=best_sparse,
            result3=result3,
            dense_ply=dense_ply,  # P0 突破口 1：傳遞密集點雲路徑
        )
        print(f"[OK] Agent contract 已導出：{contract_paths['local_contract']}")
        print(f"[OK] Agent event 已導出：{contract_paths['event_file']}")
        
        # 計算總耗時
        elapsed_seconds = time.time() - start_time
        elapsed_minutes = elapsed_seconds / 60
        elapsed_hours = elapsed_minutes / 60
        
        if elapsed_hours >= 1:
            elapsed_str = f"{elapsed_hours:.1f} 小時"
        else:
            elapsed_str = f"{elapsed_minutes:.1f} 分鐘"
        
        # 🎉 Phase 1A 最終摘要
        print("\n" + "="*70)
        print("🎉 Phase 1A SfM 重建 - 完成")
        print("="*70)
        print(f"\n⏱️  總耗時：{elapsed_str} ({int(elapsed_seconds)} 秒)")
        print("\n📊 最終統計：")
        print(f"  ✓ Step 1 特徵提取: {result1['cameras_count']} 幀, {result1['total_features']:,} 特徵")
        print(f"  ✓ Step 2 匹配驗證: {result2['num_matches']:,} 匹配, 內點率 {result2['inlier_ratio']:.3f}")
        print(
            f"  ✓ Step 3 重建驗證: {result3['registered_images_count']} 註冊影像, "
            f"{result3['points3d_count']:,} 點雲"
        )
        print("\n🚀 下一步：")
        if result3['can_proceed_to_3dgs']:
            print("  ➜ 可以進行 3DGS 訓練 [train_3dgs.py]")
            print(f"     稀疏模型: {best_sparse.resolve()}")
            print(f"     驗證報告: {paths.reports_dir / 'pointcloud_validation_report.json'}")
        else:
            print("  ⚠️  重建失敗，無法進行 3DGS 訓練")
            print("     請檢查數據品質或調整 COLMAP 參數")
        print("="*70 + "\n")
    
    except subprocess.CalledProcessError as e:
        # 提示最常見錯誤的檢查點
        msg = (
            "\nCOLMAP 指令執行失敗。\n"
            "請檢查：\n"
            "  1) 是否真的可在終端執行 `colmap`（或你指定的 --colmap-bin 路徑是否正確）。\n"
            "  2) 影像夾內容是否正常（不是空的、影像可讀）。\n"
            "  3) 若仍失敗，先把 --use-gpu 設為 False 試試（CPU 比較穩）。\n"
            "  4) `colmap help sequential_matcher` 查看你的版本支援哪些參數。\n"
            "  5) 若 mapper_type=glomap，請確認 `glomap mapper -h` 可正常執行。\n"
        )
        raise SystemExit(msg) from e

if __name__ == "__main__":
    app()
