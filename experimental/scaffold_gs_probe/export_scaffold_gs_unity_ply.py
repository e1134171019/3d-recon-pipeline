"""Export a Scaffold-GS checkpoint as a Unity-compatible preview PLY.

Scaffold-GS stores anchors, offsets, and MLP weights. UnityGaussianSplatting
expects explicit INRIA-style Gaussian splats. This sandbox utility expands the
neural Gaussians for one reference camera and writes a *single-camera preview*
PLY that can be used to test Unity import compatibility.
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


def rgb_to_sh(rgb: np.ndarray) -> np.ndarray:
    return (rgb - 0.5) / C0


def logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x))


def write_inria_ply(
    out_path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    opacity_linear: np.ndarray,
    scaling_linear: np.ndarray,
    rotation: np.ndarray,
    *,
    unity_coordinates: bool,
) -> None:
    xyz = xyz.astype(np.float32, copy=True)
    rotation = rotation.astype(np.float32, copy=True)

    if unity_coordinates:
        xyz[:, 1] = -xyz[:, 1]
        rotation[:, 1] = -rotation[:, 1]
        rotation[:, 3] = -rotation[:, 3]

    n = xyz.shape[0]
    f_dc = rgb_to_sh(np.clip(rgb.astype(np.float32), 0.0, 1.0))
    f_rest = np.zeros((n, 45), dtype=np.float32)
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-output", default="")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--min-opacity", default=0.0, type=float)
    parser.add_argument("--max-splats", default=0, type=int)
    parser.add_argument("--no-unity-coordinates", action="store_true")
    args = get_combined_args(parser)
    return model.extract(args), args


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

        camera = cameras[args.camera_index]
        xyz, color, opacity, scaling, rotation = generate_neural_gaussians(
            camera, gaussians, visible_mask=None, is_training=False
        )

        xyz_np = xyz.detach().cpu().numpy()
        color_np = color.detach().cpu().numpy()
        opacity_np = opacity.detach().cpu().numpy().reshape(-1)
        scaling_np = scaling.detach().cpu().numpy()
        rotation_np = rotation.detach().cpu().numpy()

    keep = np.ones(opacity_np.shape[0], dtype=bool)
    if args.min_opacity > 0.0:
        keep &= opacity_np >= args.min_opacity

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
        unity_coordinates=not args.no_unity_coordinates,
    )

    report = {
        "source_model_path": dataset.model_path,
        "loaded_iteration": int(scene.loaded_iter),
        "export_mode": "single_camera_preview",
        "complete_scene": False,
        "camera_set": args.camera_set,
        "camera_index": int(args.camera_index),
        "camera_uid": int(camera.uid),
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
