from __future__ import annotations

import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def _check_python() -> bool:
    _print_header("Python")
    print(f"Executable: {sys.executable}")
    print(f"Version: {sys.version.split()[0]}")
    return True


def _check_torch() -> bool:
    _print_header("PyTorch / CUDA")
    try:
        import torch
    except Exception as exc:
        print(f"FAIL: 無法 import torch: {exc}")
        return False

    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        return True

    print("FAIL: torch 看不到 GPU")
    return False


def _check_env_vars() -> bool:
    _print_header("Environment Variables")
    cuda_home = os.environ.get("CUDA_HOME", "<unset>")
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "<unset>")

    print(f"CUDA_HOME: {cuda_home}")
    print(f"TORCH_CUDA_ARCH_LIST: {arch_list}")

    if arch_list == "12.0":
        print("FAIL: 偵測到舊的 TORCH_CUDA_ARCH_LIST=12.0，這是目前已知的風險配置")
        return False

    return True


def _check_numpy() -> bool:
    _print_header("NumPy")
    try:
        import numpy as np
    except Exception as exc:
        print(f"FAIL: 無法 import numpy: {exc}")
        return False

    print(f"numpy: {np.__version__}")
    return True


def _check_gsplat() -> bool:
    _print_header("gsplat")
    try:
        import gsplat
    except Exception as exc:
        print(f"FAIL: 無法 import gsplat: {exc}")
        return False

    print(f"gsplat: {gsplat.__version__}")
    return True


def _check_runtime_deps() -> bool:
    _print_header("Trainer Runtime Dependencies")
    required = [
        ("typer", "typer"),
        ("rich", "rich"),
        ("yaml", "PyYAML"),
        ("imageio", "imageio"),
        ("viser", "viser"),
        ("nerfview", "nerfview"),
        ("torchmetrics", "torchmetrics"),
        ("lpips", "lpips"),
    ]

    failed = False
    for module_name, label in required:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "<no __version__>")
            print(f"OK   {label}: {version}")
        except Exception as exc:
            failed = True
            print(f"FAIL {label}: {exc}")

    return not failed


def _check_pycolmap() -> bool:
    _print_header("pycolmap")
    try:
        import pycolmap  # noqa: F401
    except Exception as exc:
        print(f"WARN: 無法 import pycolmap: {exc}")
        print("INFO: SfM 主線仍可執行，但統計讀取會退回較保守的 fallback")
        return True

    print("OK: pycolmap 可用")
    return True


def _check_colmap() -> bool:
    _print_header("COLMAP")
    colmap_exe = PROJECT_ROOT / "colmap" / "bin" / "colmap.exe"
    print(f"Expected path: {colmap_exe}")
    if colmap_exe.exists():
        print("OK: 找到 repo 內 COLMAP")
        return True

    print("FAIL: 找不到 repo 內 COLMAP")
    return False


def _check_reports() -> bool:
    _print_header("Pipeline Artifacts")
    report_path = PROJECT_ROOT / "outputs" / "reports" / "pointcloud_validation_report.json"
    if not report_path.exists():
        print("INFO: 尚未找到 SfM 驗證報告，這在純環境檢查階段是可接受的")
        return True

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"FAIL: 驗證報告存在但無法讀取: {exc}")
        return False

    print(f"Report: {report_path}")
    print(f"can_proceed_to_3dgs: {report.get('can_proceed_to_3dgs')}")
    print(f"images_count: {report.get('images_count')}")
    print(f"points3d_count: {report.get('points3d_count')}")
    return True


def main() -> int:
    print("3D Recon Pipeline Smoke Test")
    print(f"Project root: {PROJECT_ROOT}")

    checks = [
        ("python", _check_python()),
        ("env_vars", _check_env_vars()),
        ("torch_cuda", _check_torch()),
        ("numpy", _check_numpy()),
        ("gsplat", _check_gsplat()),
        ("trainer_runtime", _check_runtime_deps()),
        ("pycolmap", _check_pycolmap()),
        ("colmap", _check_colmap()),
        ("pipeline_artifacts", _check_reports()),
    ]

    failed = [name for name, ok in checks if not ok]

    _print_header("Summary")
    for name, ok in checks:
        print(f"{'OK  ' if ok else 'FAIL'} {name}")

    if failed:
        print("\nSmoke test failed.")
        return 1

    print("\nSmoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
