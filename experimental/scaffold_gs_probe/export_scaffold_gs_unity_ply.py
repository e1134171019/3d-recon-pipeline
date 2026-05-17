"""Export a Scaffold-GS checkpoint as a Unity-compatible preview PLY.

Scaffold-GS stores anchors, offsets, and MLP weights. UnityGaussianSplatting
expects explicit INRIA-style Gaussian splats. This sandbox utility expands the
neural Gaussians for one reference camera and can optionally:

- average the view-dependent color branch over N sampled cameras
- require visibility in at least N sampled cameras
- fit sampled MLP colors into degree-1/3 spherical harmonics at export time
"""

from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
SCAFFOLD_REPO = SCRIPT_ROOT / "repo" / "Scaffold-GS"
if not SCAFFOLD_REPO.exists():
    raise SystemExit(f"Scaffold-GS repo not found: {SCAFFOLD_REPO}")

sys.path.insert(0, str(SCAFFOLD_REPO))
os.chdir(SCAFFOLD_REPO)

import torch  # noqa: E402
from arguments import ModelParams, get_combined_args  # noqa: E402
from gaussian_renderer import generate_neural_gaussians  # noqa: E402
from scene import Scene  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402


C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)
Y_ODD_SH_INDICES = (1, 4, 5, 9, 10, 11)


def rgb_to_sh(rgb: np.ndarray) -> np.ndarray:
    return (rgb - 0.5) / C0


def logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x))


def sh_basis(dirs: np.ndarray, degree: int) -> np.ndarray:
    if degree < 0 or degree > 3:
        raise ValueError("degree must be within [0, 3]")
    if dirs.ndim != 2 or dirs.shape[1] != 3:
        raise ValueError("dirs must have shape [N, 3]")

    x = dirs[:, 0]
    y = dirs[:, 1]
    z = dirs[:, 2]

    basis = np.empty((dirs.shape[0], (degree + 1) ** 2), dtype=np.float32)
    basis[:, 0] = C0
    if degree >= 1:
        basis[:, 1] = -C1 * y
        basis[:, 2] = C1 * z
        basis[:, 3] = -C1 * x
    if degree >= 2:
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        yz = y * z
        xz = x * z
        basis[:, 4] = C2[0] * xy
        basis[:, 5] = C2[1] * yz
        basis[:, 6] = C2[2] * (2.0 * zz - xx - yy)
        basis[:, 7] = C2[3] * xz
        basis[:, 8] = C2[4] * (xx - yy)
    if degree >= 3:
        basis[:, 9] = C3[0] * y * (3.0 * x * x - y * y)
        basis[:, 10] = C3[1] * x * y * z
        basis[:, 11] = C3[2] * y * (4.0 * z * z - x * x - y * y)
        basis[:, 12] = C3[3] * z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y)
        basis[:, 13] = C3[4] * x * (4.0 * z * z - x * x - y * y)
        basis[:, 14] = C3[5] * z * (x * x - y * y)
        basis[:, 15] = C3[6] * x * (x * x - 3.0 * y * y)
    return basis


def fit_sh_coefficients(
    sample_dirs: np.ndarray,
    sample_colors: np.ndarray,
    sample_hits: np.ndarray,
    fallback_rgb: np.ndarray,
    degree: int,
    *,
    regularization: float = 1e-4,
    chunk_size: int = 4096,
) -> tuple[np.ndarray, int]:
    coeff_count = (degree + 1) ** 2
    sh_coeffs = np.zeros((fallback_rgb.shape[0], 16, 3), dtype=np.float32)
    sh_coeffs[:, 0, :] = rgb_to_sh(np.clip(fallback_rgb.astype(np.float32), 0.0, 1.0))

    hit_counts = sample_hits.sum(axis=1)
    fitted_mask = hit_counts > 0
    fitted_count = int(fitted_mask.sum())
    if fitted_count == 0:
        return sh_coeffs, fitted_count

    eye = np.eye(coeff_count, dtype=np.float32) * regularization
    centered_colors = np.clip(sample_colors.astype(np.float32), 0.0, 1.0) - 0.5
    valid_indices = np.flatnonzero(fitted_mask)

    for start in range(0, valid_indices.shape[0], chunk_size):
        chunk = valid_indices[start : start + chunk_size]
        dirs_chunk = sample_dirs[chunk]
        colors_chunk = centered_colors[chunk]
        hits_chunk = sample_hits[chunk].astype(np.float32)

        dirs_flat = dirs_chunk.reshape(-1, 3)
        basis_chunk = sh_basis(dirs_flat, degree).reshape(
            chunk.shape[0], sample_dirs.shape[1], coeff_count
        )
        weighted_basis = basis_chunk * hits_chunk[..., None]
        ata = np.einsum("bnc,bnd->bcd", weighted_basis, basis_chunk) + eye[None, :, :]
        aty = np.einsum("bnc,bnk->bck", weighted_basis, colors_chunk)
        coeffs_chunk = np.linalg.solve(ata, aty)
        sh_coeffs[chunk, :coeff_count, :] = coeffs_chunk

    return sh_coeffs, fitted_count


def write_inria_ply(
    out_path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    opacity_linear: np.ndarray,
    scaling_linear: np.ndarray,
    rotation: np.ndarray,
    *,
    sh_coefficients: np.ndarray | None,
    unity_coordinates: bool,
) -> None:
    xyz = xyz.astype(np.float32, copy=True)
    rotation = rotation.astype(np.float32, copy=True)

    if unity_coordinates:
        xyz[:, 1] = -xyz[:, 1]
        rotation[:, 1] = -rotation[:, 1]
        rotation[:, 3] = -rotation[:, 3]

    n = xyz.shape[0]
    if sh_coefficients is None:
        sh_coefficients = np.zeros((n, 16, 3), dtype=np.float32)
        sh_coefficients[:, 0, :] = rgb_to_sh(np.clip(rgb.astype(np.float32), 0.0, 1.0))
    else:
        sh_coefficients = sh_coefficients.astype(np.float32, copy=True)
        if unity_coordinates:
            for sh_index in Y_ODD_SH_INDICES:
                if sh_index < sh_coefficients.shape[1]:
                    sh_coefficients[:, sh_index, :] *= -1.0

    f_dc = sh_coefficients[:, 0, :]
    f_rest = sh_coefficients[:, 1:, :].transpose(0, 2, 1).reshape(n, -1)
    opacities = logit(opacity_linear.reshape(-1).astype(np.float32))
    scales = np.log(np.clip(scaling_linear.astype(np.float32), 1e-8, None))

    fields: list[tuple[str, np.ndarray]] = []
    for idx, name in enumerate(("x", "y", "z")):
        fields.append((name, xyz[:, idx]))
    fields.extend(
        [
            ("nx", np.zeros(n, dtype=np.float32)),
            ("ny", np.zeros(n, dtype=np.float32)),
            ("nz", np.zeros(n, dtype=np.float32)),
        ]
    )
    for idx in range(3):
        fields.append((f"f_dc_{idx}", f_dc[:, idx]))
    for idx in range(45):
        fields.append((f"f_rest_{idx}", f_rest[:, idx]))
    fields.append(("opacity", opacities))
    for idx in range(3):
        fields.append((f"scale_{idx}", scales[:, idx]))
    for idx in range(4):
        fields.append((f"rot_{idx}", rotation[:, idx]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fh:
        header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
        header.extend(f"property float {name}" for name, _ in fields)
        header.append("end_header")
        fh.write(("\n".join(header) + "\n").encode("ascii"))

        arr = np.zeros(n, dtype=[(name, np.float32) for name, _ in fields])
        for name, values in fields:
            arr[name] = values.astype(np.float32)
        arr.tofile(fh)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--camera-set", choices=("test", "train"), default="test")
    parser.add_argument("--camera-index", default=0, type=int)
    parser.add_argument("--avg-camera-set", choices=("same", "test", "train"), default="same")
    parser.add_argument("--avg-camera-count", default=1, type=int)
    parser.add_argument("--visibility-filter-n", default=0, type=int)
    parser.add_argument("--sh-fit-degree", choices=(0, 1, 2, 3), default=0, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-output", default="")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--min-opacity", default=0.0, type=float)
    parser.add_argument("--max-splats", default=0, type=int)
    parser.add_argument("--no-unity-coordinates", action="store_true")
    args = get_combined_args(parser)
    return model.extract(args), args


def sampled_camera_indices(camera_count: int, requested_count: int) -> list[int]:
    if camera_count <= 0:
        return []
    if requested_count <= 0 or requested_count >= camera_count:
        return list(range(camera_count))

    samples = np.linspace(0, camera_count - 1, num=requested_count)
    picked: list[int] = []
    seen: set[int] = set()
    for value in samples:
        idx = int(round(float(value)))
        idx = max(0, min(camera_count - 1, idx))
        if idx not in seen:
            picked.append(idx)
            seen.add(idx)

    if len(picked) < requested_count:
        for idx in range(camera_count):
            if idx not in seen:
                picked.append(idx)
                seen.add(idx)
            if len(picked) >= requested_count:
                break
    return picked


def main() -> None:
    dataset, args = parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))

    with torch.no_grad():
        gaussians = GaussianModel(
            dataset.feat_dim,
            dataset.n_offsets,
            dataset.voxel_size,
            dataset.update_depth,
            dataset.update_init_factor,
            dataset.update_hierachy_factor,
            dataset.use_feat_bank,
            dataset.appearance_dim,
            dataset.ratio,
            dataset.add_opacity_dist,
            dataset.add_cov_dist,
            dataset.add_color_dist,
        )
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        gaussians.eval()

        cameras = (
            scene.getTestCameras()
            if args.camera_set == "test"
            else scene.getTrainCameras()
        )
        if not cameras:
            raise SystemExit(f"No {args.camera_set} cameras available")
        if args.camera_index < 0 or args.camera_index >= len(cameras):
            raise SystemExit(
                f"camera-index {args.camera_index} out of range for {len(cameras)} cameras"
            )

        reference_camera = cameras[args.camera_index]
        (
            xyz,
            color,
            opacity,
            scaling,
            rotation,
            _reference_neural_opacity,
            reference_mask,
        ) = generate_neural_gaussians(
            reference_camera, gaussians, visible_mask=None, is_training=True
        )

        xyz_np = xyz.detach().cpu().numpy()
        color_np = color.detach().cpu().numpy()
        opacity_np = opacity.detach().cpu().numpy().reshape(-1)
        scaling_np = scaling.detach().cpu().numpy()
        rotation_np = rotation.detach().cpu().numpy()
        reference_mask_np = reference_mask.detach().cpu().numpy().astype(bool, copy=False)
        reference_flat_indices = np.flatnonzero(reference_mask_np)
        reference_anchor_indices = reference_flat_indices // gaussians.n_offsets
        reference_anchor_np = (
            gaussians.get_anchor.detach().cpu().numpy()[reference_anchor_indices]
            if reference_flat_indices.size
            else np.zeros((0, 3), dtype=np.float32)
        )

        if args.avg_camera_set == "same":
            averaging_camera_set = args.camera_set
            averaging_cameras = cameras
        else:
            averaging_camera_set = args.avg_camera_set
            averaging_cameras = (
                scene.getTestCameras()
                if averaging_camera_set == "test"
                else scene.getTrainCameras()
            )
        if not averaging_cameras:
            raise SystemExit(f"No {averaging_camera_set} cameras available for averaging")

        averaging_indices = sampled_camera_indices(len(averaging_cameras), args.avg_camera_count)
        if not averaging_indices:
            raise SystemExit("No averaging cameras were selected")

        is_multi_camera_average = not (
            len(averaging_indices) == 1
            and averaging_camera_set == args.camera_set
            and averaging_indices[0] == args.camera_index
        )
        need_per_camera_samples = args.visibility_filter_n > 0 or args.sh_fit_degree > 0
        fallback_splats = 0
        averaging_camera_uids = [int(averaging_cameras[idx].uid) for idx in averaging_indices]
        visibility_counts: np.ndarray | None = None
        sh_coefficients: np.ndarray | None = None
        sh_fitted_splats = 0

        if is_multi_camera_average or need_per_camera_samples:
            color_sum = np.zeros_like(color_np, dtype=np.float32)
            color_hits = np.zeros(reference_flat_indices.shape[0], dtype=np.int32)
            total_slots = int(reference_mask_np.shape[0])
            sample_hits = None
            sample_colors = None
            sample_dirs = None

            if need_per_camera_samples:
                sample_hits = np.zeros(
                    (reference_flat_indices.shape[0], len(averaging_indices)), dtype=bool
                )
                sample_colors = np.zeros(
                    (reference_flat_indices.shape[0], len(averaging_indices), 3),
                    dtype=np.float32,
                )
                sample_dirs = np.zeros_like(sample_colors)

            for sample_idx, avg_idx in enumerate(averaging_indices):
                avg_camera = averaging_cameras[avg_idx]
                (
                    _avg_xyz,
                    avg_color,
                    _avg_opacity,
                    _avg_scaling,
                    _avg_rotation,
                    _avg_neural_opacity,
                    avg_mask,
                ) = generate_neural_gaussians(
                    avg_camera, gaussians, visible_mask=None, is_training=True
                )

                avg_mask_np = avg_mask.detach().cpu().numpy().astype(bool, copy=False)
                full_color = np.zeros((total_slots, 3), dtype=np.float32)
                full_color[avg_mask_np] = avg_color.detach().cpu().numpy()

                contributing = avg_mask_np[reference_flat_indices]
                if np.any(contributing):
                    selected_indices = reference_flat_indices[contributing]
                    selected_colors = full_color[selected_indices]
                    color_sum[contributing] += selected_colors
                    color_hits[contributing] += 1

                    if need_per_camera_samples:
                        assert sample_hits is not None
                        assert sample_colors is not None
                        assert sample_dirs is not None
                        sample_hits[contributing, sample_idx] = True
                        sample_colors[contributing, sample_idx] = selected_colors
                        camera_center = avg_camera.camera_center.detach().cpu().numpy()
                        dirs = reference_anchor_np[contributing] - camera_center[None, :]
                        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
                        norms = np.clip(norms, 1e-8, None)
                        sample_dirs[contributing, sample_idx] = dirs / norms

            visibility_counts = color_hits.astype(np.int32, copy=True)
            if is_multi_camera_average:
                averaged_color = color_np.copy()
                valid_hits = color_hits > 0
                if np.any(valid_hits):
                    averaged_color[valid_hits] = color_sum[valid_hits] / color_hits[valid_hits, None]
                fallback_splats = int((~valid_hits).sum())
                color_np = averaged_color

            if args.sh_fit_degree > 0 and sample_hits is not None and sample_colors is not None and sample_dirs is not None:
                sh_coefficients, sh_fitted_splats = fit_sh_coefficients(
                    sample_dirs,
                    sample_colors,
                    sample_hits,
                    color_np,
                    args.sh_fit_degree,
                )

    keep = np.ones(opacity_np.shape[0], dtype=bool)
    if args.min_opacity > 0.0:
        keep &= opacity_np >= args.min_opacity
    if args.visibility_filter_n > 0:
        if visibility_counts is None:
            raise SystemExit("visibility-filter-n requires averaging/sample cameras to be evaluated")
        keep &= visibility_counts >= args.visibility_filter_n

    if args.max_splats > 0 and keep.sum() > args.max_splats:
        kept_indices = np.flatnonzero(keep)
        order = np.argsort(-opacity_np[kept_indices])[: args.max_splats]
        selected = kept_indices[order]
        keep = np.zeros_like(keep)
        keep[selected] = True

    out_path = Path(args.output).resolve()
    write_inria_ply(
        out_path,
        xyz_np[keep],
        color_np[keep],
        opacity_np[keep],
        scaling_np[keep],
        rotation_np[keep],
        sh_coefficients=sh_coefficients[keep] if sh_coefficients is not None else None,
        unity_coordinates=not args.no_unity_coordinates,
    )

    if args.sh_fit_degree > 0:
        export_mode = (
            f"{'multi' if len(averaging_indices) > 1 else 'single'}_camera_sh_fit_degree_{args.sh_fit_degree}_preview"
        )
    elif is_multi_camera_average:
        export_mode = "multi_camera_color_average_preview"
    else:
        export_mode = "single_camera_preview"

    report = {
        "source_model_path": dataset.model_path,
        "loaded_iteration": int(scene.loaded_iter),
        "export_mode": export_mode,
        "complete_scene": False,
        "camera_set": args.camera_set,
        "camera_index": int(args.camera_index),
        "camera_uid": int(reference_camera.uid),
        "reference_camera_set": args.camera_set,
        "reference_camera_index": int(args.camera_index),
        "reference_camera_uid": int(reference_camera.uid),
        "averaging_camera_set": averaging_camera_set,
        "averaging_camera_count_requested": int(args.avg_camera_count),
        "averaging_camera_count_used": len(averaging_indices),
        "averaging_camera_indices": averaging_indices,
        "averaging_camera_uids": averaging_camera_uids,
        "averaged_channels": ["color"] if is_multi_camera_average else [],
        "color_average_fallback_splats": fallback_splats,
        "visibility_filter_n": int(args.visibility_filter_n),
        "visibility_hits_min": int(visibility_counts.min()) if visibility_counts is not None and visibility_counts.size else 0,
        "visibility_hits_max": int(visibility_counts.max()) if visibility_counts is not None and visibility_counts.size else 0,
        "visibility_hits_mean": round(float(visibility_counts.mean()), 4) if visibility_counts is not None and visibility_counts.size else 0.0,
        "sh_fit_degree": int(args.sh_fit_degree),
        "sh_fit_sample_count": len(averaging_indices) if args.sh_fit_degree > 0 else 0,
        "sh_fitted_splats": int(sh_fitted_splats),
        "candidate_splats": int(opacity_np.shape[0]),
        "exported_splats": int(keep.sum()),
        "min_opacity": float(args.min_opacity),
        "max_splats": int(args.max_splats),
        "unity_coordinates": not args.no_unity_coordinates,
        "output": str(out_path),
        "output_size_mb": round(out_path.stat().st_size / (1024 * 1024), 3),
    }
    if args.report_output:
        report_path = Path(args.report_output).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
