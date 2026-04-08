"""
export_ply.py  —  從 gsplat checkpoint 匯出標準 3DGS .ply 檔案

用法：
    python src/export_ply.py
    python src/export_ply.py --ckpt outputs/3DGS_models/ckpts/ckpt_29999_rank0.pt
    python src/export_ply.py --out  outputs/3DGS_models/ply/point_cloud_final.ply
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="gsplat checkpoint → .ply")
    parser.add_argument(
        "--ckpt",
        default="outputs/3DGS_models/ckpts/ckpt_29999_rank0.pt",
        help="checkpoint 路徑",
    )
    parser.add_argument(
        "--out",
        default="outputs/3DGS_models/ply/point_cloud_final.ply",
        help="輸出 .ply 路徑",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    out_path  = Path(args.out)

    if not ckpt_path.exists():
        print(f"[ERROR] 找不到 checkpoint：{ckpt_path}")
        raise SystemExit(1)

    print(f"載入 checkpoint：{ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    splats   = ckpt["splats"]
    step     = ckpt.get("step", -1)
    print(f"  step = {step}")
    print(f"  Gaussian 數量 = {splats['means'].shape[0]:,}")
    print(f"  keys = {list(splats.keys())}")

    # 嘗試使用 gsplat 的 export_splats 函數
    try:
        from gsplat.utils import export_splats
        out_path.parent.mkdir(parents=True, exist_ok=True)
        export_splats(
            means      = splats["means"],
            scales     = splats["scales"],
            quats      = splats["quats"],
            opacities  = splats["opacities"],
            sh0        = splats["sh0"],
            shN        = splats["shN"],
            format     = "ply",
            save_to    = str(out_path),
        )
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"[OK] 匯出完成：{out_path}  ({size_mb:.1f} MB)")
    except ImportError:
        # 若 gsplat.utils.export_splats 不存在，手動寫入 PLY
        print("gsplat.utils.export_splats 不可用，改用手動寫入...")
        _write_ply_manual(splats, out_path)


def _write_ply_manual(splats: dict, out_path: Path):
    """手動將 Gaussian 參數寫入標準 3DGS PLY 格式。"""
    import numpy as np
    import struct

    means     = splats["means"].float().cpu().numpy()       # (N,3)
    scales    = splats["scales"].float().cpu().numpy()      # (N,3)
    quats     = splats["quats"].float().cpu().numpy()       # (N,4) xyzw
    opacities = splats["opacities"].float().cpu().numpy()   # (N,) logit
    sh0       = splats["sh0"].float().cpu().numpy()         # (N,1,3)
    shN       = splats["shN"].float().cpu().numpy()         # (N,K,3)

    N = means.shape[0]
    # sh0 是 DC 分量；flatten 成 f_dc_0..2
    f_dc = sh0.reshape(N, 3)                                # (N,3)
    # shN 是高階 SH；flatten
    sh_rest = shN.reshape(N, -1)                            # (N, K*3)

    print(f"  N={N:,}, sh_rest channels={sh_rest.shape[1]}")

    # 組裝欄位清單（符合 inria/3dgs，SuperSplat 等可讀）
    fields = []
    # position
    for c in ("x", "y", "z"):
        fields.append((c, means[:, ["x","y","z"].index(c)]))
    fields.append(("nx", np.zeros(N, np.float32)))
    fields.append(("ny", np.zeros(N, np.float32)))
    fields.append(("nz", np.zeros(N, np.float32)))
    for i in range(3):
        fields.append((f"f_dc_{i}", f_dc[:, i]))
    for i in range(sh_rest.shape[1]):
        fields.append((f"f_rest_{i}", sh_rest[:, i]))
    fields.append(("opacity", opacities))
    for i in range(3):
        fields.append((f"scale_{i}", scales[:, i]))
    # quats: gsplat stores (w,x,y,z); inria format is (w,x,y,z) rot_0..3
    for i, c in enumerate(("rot_0", "rot_1", "rot_2", "rot_3")):
        fields.append((c, quats[:, i]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        # ASCII header
        header_lines = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {N}",
        ]
        for name, _ in fields:
            header_lines.append(f"property float {name}")
        header_lines.append("end_header")
        header_str = "\n".join(header_lines) + "\n"
        f.write(header_str.encode("ascii"))
        # Binary data
        dtype_list = [(name, np.float32) for name, _ in fields]
        arr = np.zeros(N, dtype=dtype_list)
        for name, data in fields:
            arr[name] = data.astype(np.float32)
        arr.tofile(f)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[OK] 手動 PLY 寫入完成：{out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
