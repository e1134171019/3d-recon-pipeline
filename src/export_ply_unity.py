"""
export_ply_unity.py  —  從 gsplat checkpoint 匯出修正座標的 .ply（給 Unity 用）

主要修正：
  1. 反正規化（inverse normalize transform）：還原到 COLMAP 世界座標
     - 位置：means = inv_T @ means
     - 旋轉：quats = inv_R_quat * quats（quaternion 乘法）
     - 縮放：scales -= log(uniform_scale)

用法：
    python -m src.export_ply_unity
    python -m src.export_ply_unity --ckpt outputs/3DGS_models/ckpts/ckpt_29999_rank0.pt
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from src.utils.agent_contracts import infer_outputs_root, trigger_decision_layer, write_stage_contract


# ─── Quaternion helpers (wxyz convention) ────────────────────────────

def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → quaternion (w, x, y, z)."""
    # Shepperd's method
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Quaternion multiply: q1 * q2 (Hamilton product).
    q1: (4,) single quaternion (wxyz)
    q2: (N, 4) array of quaternions (wxyz)
    Returns: (N, 4) array
    """
    w1, x1, y1, z1 = q1
    w2 = q2[:, 0]; x2 = q2[:, 1]; y2 = q2[:, 2]; z2 = q2[:, 3]

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    out = np.stack([w, x, y, z], axis=-1)
    # Normalize
    norms = np.linalg.norm(out, axis=-1, keepdims=True)
    return out / np.maximum(norms, 1e-12)


def _reconstruct_normalize_transform(data_dir: str) -> np.ndarray:
    """
    從 COLMAP 場景重建 gsplat 訓練時使用的 normalize transform。
    回傳 4x4 transform matrix（與 gsplat_runner/datasets/colmap.py 完全等價）。
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / "gsplat_runner"))
    from datasets.colmap import Parser

    parser = Parser(
        data_dir=data_dir,
        factor=1,
        normalize=True,
        test_every=8,
    )
    return parser.transform  # np.ndarray, (4, 4)


def denormalize_splats(means, quats, scales, transform):
    """
    將正規化空間的 splat 資料轉回原始 COLMAP 空間。

    transform: 正規化時使用的 4x4 矩陣 (含 rotation * scale + translation)
    """
    inv = np.linalg.inv(transform)

    # ── 1. 反正規化位置 ──
    means_out = means @ inv[:3, :3].T + inv[:3, 3]

    # ── 2. 反正規化 Gaussian scales ──
    # transform[:3,:3] = scale * R → 提取 uniform scale
    R_with_scale = transform[:3, :3]
    col_norms = np.linalg.norm(R_with_scale, axis=0)
    uniform_scale = col_norms.mean()  # should be same for all 3 columns
    # Gaussian scale 在 log space → 減去 log(uniform_scale)
    scales_out = scales - np.log(uniform_scale)

    # ── 3. 反正規化 quaternion ──
    # 提取純旋轉矩陣（去掉 scale）
    R_pure = R_with_scale / uniform_scale
    # 逆旋轉的 quaternion
    R_inv = R_pure.T  # R 是正交矩陣，逆 = 轉置
    q_inv = rotmat_to_quat(R_inv)
    print(f"  逆旋轉 quaternion: {q_inv}")
    # 每個 Gaussian 的 quat 左乘 q_inv：q_new = q_inv * q_old
    quats_out = quat_multiply(q_inv, quats)

    return means_out, quats_out, scales_out


def _apply_export_filters(
    means: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    opacities: np.ndarray,
    sh0: np.ndarray,
    shN: np.ndarray,
    *,
    min_opacity: float = 0.0,
    max_scale: float = 0.0,
    max_scale_percentile: float = 0.0,
):
    """
    Apply export-side splat filters and return filtered arrays plus metadata.

    Percentile-based scale filtering is more stable than a hard absolute
    threshold across different runs because the linear scale distribution can
    drift with normalization and training dynamics.
    """
    keep_mask = np.ones(means.shape[0], dtype=bool)
    opacity_linear = 1.0 / (1.0 + np.exp(-opacities))
    linear_scales = np.exp(scales)
    max_linear_scale = linear_scales.max(axis=1)

    effective_max_scale = max_scale
    if max_scale_percentile > 0.0:
        if not 0.0 < max_scale_percentile <= 100.0:
            raise ValueError("max_scale_percentile must be within (0, 100].")
        percentile_threshold = float(np.percentile(max_linear_scale, max_scale_percentile))
        if effective_max_scale > 0.0:
            effective_max_scale = min(effective_max_scale, percentile_threshold)
        else:
            effective_max_scale = percentile_threshold
    else:
        percentile_threshold = 0.0

    if min_opacity > 0.0:
        keep_mask &= opacity_linear >= min_opacity
        print(
            f"[FILTER] min_opacity={min_opacity:.4f} -> 保留 {int(keep_mask.sum()):,}/{len(keep_mask):,}"
        )
    if effective_max_scale > 0.0:
        keep_mask &= max_linear_scale <= effective_max_scale
        if max_scale_percentile > 0.0:
            print(
                f"[FILTER] max_scale_percentile<={max_scale_percentile:.1f}"
                f" (threshold={effective_max_scale:.4f}) -> 保留 {int(keep_mask.sum()):,}/{len(keep_mask):,}"
            )
        else:
            print(
                f"[FILTER] max_scale<={effective_max_scale:.4f} -> 保留 {int(keep_mask.sum()):,}/{len(keep_mask):,}"
            )
    if not np.all(keep_mask):
        means = means[keep_mask]
        scales = scales[keep_mask]
        quats = quats[keep_mask]
        opacities = opacities[keep_mask]
        sh0 = sh0[keep_mask]
        shN = shN[keep_mask]
        print(f"[FILTER] 最終保留 splats: {means.shape[0]:,}")

    metadata = {
        "input_splats": int(len(keep_mask)),
        "kept_splats": int(keep_mask.sum()),
        "dropped_splats": int(len(keep_mask) - keep_mask.sum()),
        "min_opacity": float(min_opacity),
        "max_scale": float(max_scale),
        "max_scale_percentile": float(max_scale_percentile),
        "effective_max_scale": float(effective_max_scale),
        "opacity_min_linear": float(opacity_linear.min()) if len(opacity_linear) else 0.0,
        "opacity_max_linear": float(opacity_linear.max()) if len(opacity_linear) else 0.0,
        "scale_percentile_threshold": float(percentile_threshold),
    }
    return means, scales, quats, opacities, sh0, shN, metadata


def main():
    parser = argparse.ArgumentParser(description="gsplat checkpoint → Unity .ply")
    parser.add_argument(
        "--ckpt",
        default="outputs/3DGS_models/ckpts/ckpt_29999_rank0.pt",
        help="checkpoint 路徑",
    )
    parser.add_argument(
        "--data-dir",
        default="data/colmap_scene",
        help="COLMAP scene 目錄（訓練時用的 data_dir）",
    )
    parser.add_argument(
        "--out",
        default="outputs/3DGS_models/ply/point_cloud_unity.ply",
        help="輸出 .ply 路徑",
    )
    parser.add_argument(
        "--no-denormalize",
        action="store_true",
        help="跳過反正規化（直接輸出正規化座標）",
    )
    parser.add_argument(
        "--unity",
        action="store_true",
        help="Apply Unity coordinate handedness conversion (Y-flip)",
    )
    parser.add_argument(
        "--min-opacity",
        type=float,
        default=0.0,
        help="Keep only splats whose sigmoid(opacity) is >= this threshold",
    )
    parser.add_argument(
        "--max-scale",
        type=float,
        default=0.0,
        help="Keep only splats whose max(linear scale) is <= this threshold; 0 disables",
    )
    parser.add_argument(
        "--max-scale-percentile",
        type=float,
        default=0.0,
        help="Keep only splats whose max(linear scale) is within this percentile; 0 disables",
    )
    parser.add_argument(
        "--params-json",
        default="",
        help="Optional JSON generated by agent layer; fills min_opacity/max_scale if CLI flags are unset",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    out_path = Path(args.out)

    if not ckpt_path.exists():
        print(f"[ERROR] 找不到 checkpoint：{ckpt_path}")
        raise SystemExit(1)

    if args.params_json:
        params_path = Path(args.params_json)
        if not params_path.exists():
            print(f"[ERROR] 找不到 params json：{params_path}")
            raise SystemExit(1)
        with open(params_path, "r", encoding="utf-8") as f:
            params_payload = json.load(f)
        recommended = params_payload.get("recommended_params", {})
        if args.min_opacity <= 0.0:
            args.min_opacity = float(recommended.get("min_opacity", 0.0))
        if args.max_scale <= 0.0:
            args.max_scale = float(recommended.get("max_scale", 0.0))
        if args.max_scale_percentile <= 0.0:
            args.max_scale_percentile = float(recommended.get("max_scale_percentile", 0.0))
        print(
            f"[INFO] 已載入參數檔 {params_path} -> "
            f"min_opacity={args.min_opacity:.4f}, max_scale={args.max_scale:.4f}, "
            f"max_scale_percentile={args.max_scale_percentile:.1f}"
        )

    # ── 載入 checkpoint ──────────────────────
    print(f"載入 checkpoint：{ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    splats = ckpt["splats"]
    step = ckpt.get("step", -1)
    print(f"  step = {step}")
    print(f"  Gaussian 數量 = {splats['means'].shape[0]:,}")

    means     = splats["means"].float().cpu().numpy()       # (N, 3)
    scales    = splats["scales"].float().cpu().numpy()      # (N, 3)
    quats     = splats["quats"].float().cpu().numpy()       # (N, 4) wxyz
    opacities = splats["opacities"].float().cpu().numpy()   # (N,) logit
    sh0       = splats["sh0"].float().cpu().numpy()         # (N, 1, 3)
    shN       = splats["shN"].float().cpu().numpy()         # (N, K, 3)

    print(f"  正規化空間位置範圍: "
          f"X=[{means[:,0].min():.2f}, {means[:,0].max():.2f}] "
          f"Y=[{means[:,1].min():.2f}, {means[:,1].max():.2f}] "
          f"Z=[{means[:,2].min():.2f}, {means[:,2].max():.2f}]")

    # ── 反正規化 ──────────────────────────
    if not args.no_denormalize:
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"[ERROR] 找不到 COLMAP 場景目錄：{data_dir}")
            print("  若要跳過反正規化，請使用 --no-denormalize")
            raise SystemExit(1)

        print(f"重建正規化 transform 從 {data_dir} ...")
        transform = _reconstruct_normalize_transform(str(data_dir))
        print(f"  transform =\n{transform}")

        print("反正規化 means, quats, scales ...")
        means, quats, scales = denormalize_splats(means, quats, scales, transform)

        print(f"  原始空間位置範圍: "
              f"X=[{means[:,0].min():.2f}, {means[:,0].max():.2f}] "
              f"Y=[{means[:,1].min():.2f}, {means[:,1].max():.2f}] "
              f"Z=[{means[:,2].min():.2f}, {means[:,2].max():.2f}]")
    else:
        print("[INFO] 跳過反正規化")

    # ── 寫入 PLY ──────────────────────────
    if args.unity:
        print("[INFO] 套用 Unity 座標系轉換（COLMAP Right-handed Y-down -> Unity Left-handed Y-up）")
        # 1. 位置 Y 軸翻轉
        means[:, 1] = -means[:, 1]
        
        # 2. 旋轉 Quaternion 手性轉換
        # R_unity = T @ R_colmap @ T, where T = diag(1, -1, 1)
        # 經過數學推導，這等同於對 \text{quat} 的 X 和 Z 分量加上負號
        # COLMAP/gsplat: [w, x, y, z] -> [w, -x, y, -z]
        quats[:, 1] = -quats[:, 1]
        quats[:, 3] = -quats[:, 3]
        
        # 3. Scale 維持不變（因為是取絕對值的 scale）
        # 4. Spherical Harmonics 係數：
        #    由於 Y 方向反轉，所有 Y 奇數次方的 SH 係數需要加負號。
        #    DC (sh0) 沒差，但在嚴格的實作中 shN 的 Y dependent bands 要反轉。
        #    目前 3DGS-Unity 通常會忽略 SH 旋轉或只用顏色，為避免更複雜錯誤，先不動 SH。

    means, scales, quats, opacities, sh0, shN, filter_meta = _apply_export_filters(
        means,
        scales,
        quats,
        opacities,
        sh0,
        shN,
        min_opacity=args.min_opacity,
        max_scale=args.max_scale,
        max_scale_percentile=args.max_scale_percentile,
    )

    _write_ply(means, scales, quats, opacities, sh0, shN, out_path)
    print(f"\n[完成] PLY 已匯出至 {out_path}")

    project_root = Path(__file__).parent.parent
    outputs_root = infer_outputs_root(project_root, out_path)
    contract_paths = write_stage_contract(
        project_root=project_root,
        run_root=outputs_root,
        stage="export_complete",
        status="completed",
        artifacts={
            "checkpoint": ckpt_path,
            "data_dir": Path(args.data_dir),
            "ply_file": out_path,
            "params_json": Path(args.params_json) if args.params_json else "",
        },
        metrics={
            "num_splats": int(means.shape[0]),
            "ply_size_mb": round(out_path.stat().st_size / (1024 ** 2), 3),
            "min_opacity": args.min_opacity,
            "max_scale": args.max_scale,
            "max_scale_percentile": args.max_scale_percentile,
            "effective_max_scale": filter_meta["effective_max_scale"],
            "dropped_splats": filter_meta["dropped_splats"],
        },
        params={
            "unity": args.unity,
            "no_denormalize": args.no_denormalize,
            "min_opacity": args.min_opacity,
            "max_scale": args.max_scale,
            "max_scale_percentile": args.max_scale_percentile,
        },
        summary="PLY export complete",
    )
    print(f"[OK] Agent contract 已導出：{contract_paths['local_contract']}")
    print(f"[OK] Agent event 已導出：{contract_paths['event_file']}")
    decision_contract = contract_paths.get("latest_file") or contract_paths["event_file"]
    decision_result = trigger_decision_layer(
        project_root=project_root,
        contract_path=decision_contract,
    )
    if decision_result["status"] == "completed":
        print(f"[OK] Agent decision 已更新：{decision_result.get('decision_path', '')}")
    elif decision_result["status"] == "warning":
        print(f"[WARN] Agent decision hook 警告：{decision_result.get('reason', '') or decision_result.get('decision_path', '')}")
    else:
        print(
            f"[WARN] Agent decision hook 失敗：returncode={decision_result.get('returncode')} "
            f"{decision_result.get('stderr', '') or decision_result.get('stdout', '')}"
        )


def _write_ply(means, scales, quats, opacities, sh0, shN, out_path: Path):
    """寫入標準 3DGS PLY 格式（INRIA 相容）"""
    # 對 SH 進行轉置以符合 INRIA 標準 PLY 格式
    # INRIA 預期 `f_rest_0`..`f_rest_14` 全部都是 Red (channel 0)
    # `f_rest_15`..`f_rest_29` 全部都是 Green (channel 1)
    # `f_rest_30`..`f_rest_44` 全部都是 Blue (channel 2)
    # 所以要從 [N, K, 3] 轉成 [N, 3, K] 再 flatten 成 [N, 3*K]
    N = means.shape[0]
    f_dc = sh0.transpose(0, 2, 1).reshape(N, 3)          # (N, 3, 1) -> (N, 3)
    sh_rest = shN.transpose(0, 2, 1).reshape(N, -1)      # (N, 3, K) -> (N, 3*K)

    print(f"寫入 PLY：N={N:,}, sh_rest channels={sh_rest.shape[1]}")

    fields = []
    for c_idx, c in enumerate(("x", "y", "z")):
        fields.append((c, means[:, c_idx]))
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
    # quats: gsplat stores (w,x,y,z); INRIA format rot_0=w, rot_1=x, rot_2=y, rot_3=z
    for i in range(4):
        fields.append((f"rot_{i}", quats[:, i]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
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

        dtype_list = [(name, np.float32) for name, _ in fields]
        arr = np.zeros(N, dtype=dtype_list)
        for name, data in fields:
            arr[name] = data.astype(np.float32)
        arr.tofile(f)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[OK] PLY 寫入完成：{out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
