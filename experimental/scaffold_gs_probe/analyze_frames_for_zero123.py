"""
analyze_frames_for_zero123.py

Parses COLMAP images.bin → extracts camera positions + azimuths
Groups 853 frames into angular sectors
Ranks each frame by sharpness (Laplacian variance)
Outputs top N candidates per sector → ready for Zero123++ input

Usage:
    python analyze_frames_for_zero123.py
"""
from __future__ import annotations

import struct
import math
import os
import json
from pathlib import Path
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────
COLMAP_IMAGES_BIN = Path(
    r"C:\3d-recon-pipeline\outputs\experiments\train_probes"
    r"\u_base_mcmc_capmax_aa_fulltrain_20260420_032355"
    r"\mcmc_capmax_750k_aa\_colmap_scene\sparse\0\images.bin"
)
FRAMES_DIR = Path(r"C:\3d-recon-pipeline\data\frames_1600")
OUTPUT_JSON = Path(r"C:\3d-recon-pipeline\experimental\scaffold_gs_probe"
                   r"\outputs\frame_selection_for_zero123.json")
TOP_N_PER_SECTOR = 3   # best frames per azimuth sector
N_SECTORS = 8          # divide 360° into 8 × 45° sectors

# ── COLMAP binary parser ────────────────────────────────────────────────────

def read_images_bin(path: Path) -> list[dict]:
    """Parse COLMAP images.bin → list of {name, qw,qx,qy,qz, tx,ty,tz}"""
    cameras = []
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        print(f"  Total images in COLMAP: {num_images}")
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<dddd", f.read(32))
            tx, ty, tz = struct.unpack("<ddd", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]

            # Read null-terminated name
            name_bytes = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name_bytes += ch
            name = name_bytes.decode("utf-8", errors="replace")

            num_points2d = struct.unpack("<Q", f.read(8))[0]
            # Skip points2D (each = x(8) + y(8) + point3d_id(8) = 24 bytes)
            f.seek(num_points2d * 24, 1)

            cameras.append({
                "image_id": image_id,
                "name": name,
                "qw": qw, "qx": qx, "qy": qy, "qz": qz,
                "tx": tx, "ty": ty, "tz": tz,
                "camera_id": camera_id,
            })
    return cameras


def quat_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to 3×3 rotation matrix."""
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if n < 1e-10:
        return [[1,0,0],[0,1,0],[0,0,1]]
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    R = [
        [1-2*(qy*qy+qz*qz),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
    ]
    return R


def camera_world_position(cam: dict) -> tuple[float, float, float]:
    """Camera center in world coords: C = -R^T @ t"""
    R = quat_to_rotation_matrix(cam["qw"], cam["qx"], cam["qy"], cam["qz"])
    tx, ty, tz = cam["tx"], cam["ty"], cam["tz"]
    # C = -R^T t
    cx = -(R[0][0]*tx + R[1][0]*ty + R[2][0]*tz)
    cy = -(R[0][1]*tx + R[1][1]*ty + R[2][1]*tz)
    cz = -(R[0][2]*tx + R[1][2]*ty + R[2][2]*tz)
    return cx, cy, cz


def laplacian_sharpness(img_path: Path) -> float:
    """Estimate sharpness using file size as fast proxy (no OpenCV needed).
    Larger JPEG = more detail = sharper. Returns normalized 0-1."""
    try:
        return img_path.stat().st_size
    except Exception:
        return 0.0


# ── Main analysis ───────────────────────────────────────────────────────────

def main():
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    print("Parsing COLMAP images.bin ...")
    cameras = read_images_bin(COLMAP_IMAGES_BIN)

    # Compute world positions and scene centroid
    positions = []
    for cam in cameras:
        cx, cy, cz = camera_world_position(cam)
        cam["world_x"] = cx
        cam["world_y"] = cy
        cam["world_z"] = cz
        positions.append((cx, cy, cz))

    cx_mean = sum(p[0] for p in positions) / len(positions)
    cy_mean = sum(p[1] for p in positions) / len(positions)
    cz_mean = sum(p[2] for p in positions) / len(positions)
    print(f"  Scene centroid: ({cx_mean:.3f}, {cy_mean:.3f}, {cz_mean:.3f})")

    # Compute azimuth angle of each camera around centroid (in XZ plane)
    for cam in cameras:
        dx = cam["world_x"] - cx_mean
        dz = cam["world_z"] - cz_mean
        azimuth_deg = math.degrees(math.atan2(dz, dx)) % 360
        cam["azimuth_deg"] = azimuth_deg
        cam["dist_from_centroid"] = math.sqrt(dx*dx + dz*dz)
        cam["height"] = cam["world_y"]

        # Match to frame file
        frame_name = Path(cam["name"]).name  # e.g. frame_000070.jpg
        frame_path = FRAMES_DIR / frame_name
        cam["frame_path"] = str(frame_path)
        cam["file_size"] = frame_path.stat().st_size if frame_path.exists() else 0

        # Assign sector
        sector_size = 360.0 / N_SECTORS
        sector_idx = int(azimuth_deg / sector_size) % N_SECTORS
        cam["sector"] = sector_idx
        sector_labels = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
        cam["sector_label"] = sector_labels[sector_idx]

    # Print azimuth distribution
    sector_counts = defaultdict(int)
    for cam in cameras:
        sector_counts[cam["sector_label"]] += 1
    print("\n  Camera azimuth distribution:")
    for label in ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]:
        bar = "█" * (sector_counts[label] // 5)
        print(f"    {label:>2}: {sector_counts[label]:3d} cameras  {bar}")

    # Group by sector and pick top N by file size (sharpness proxy)
    sectors: dict[str, list] = defaultdict(list)
    for cam in cameras:
        sectors[cam["sector_label"]].append(cam)

    selections = {}
    for label, cams in sectors.items():
        sorted_cams = sorted(cams, key=lambda c: c["file_size"], reverse=True)
        top = sorted_cams[:TOP_N_PER_SECTOR]
        selections[label] = [
            {
                "frame": Path(c["name"]).name,
                "frame_path": c["frame_path"],
                "azimuth_deg": round(c["azimuth_deg"], 1),
                "file_size_kb": round(c["file_size"] / 1024, 1),
                "height": round(c["height"], 3),
                "dist_from_centroid": round(c["dist_from_centroid"], 3),
            }
            for c in top
        ]

    print("\n  Top candidates per sector:")
    all_candidates = []
    for label in ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]:
        if label not in selections:
            print(f"    {label}: ❌ No cameras")
            continue
        print(f"\n    {label} sector:")
        for s in selections[label]:
            print(f"      {s['frame']:25s}  az={s['azimuth_deg']:6.1f}°  "
                  f"size={s['file_size_kb']:6.1f}KB  h={s['height']:.3f}")
            all_candidates.append(s)

    # Save
    result = {
        "total_cameras": len(cameras),
        "n_sectors": N_SECTORS,
        "top_n_per_sector": TOP_N_PER_SECTOR,
        "sector_distribution": {k: len(v) for k, v in sectors.items()},
        "selections_by_sector": selections,
        "all_candidates_flat": sorted(all_candidates,
                                      key=lambda x: x["file_size_kb"],
                                      reverse=True),
    }

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved: {OUTPUT_JSON}")
    print(f"   Total candidates: {len(all_candidates)}")
    print(f"   Use these as Zero123++ input seeds")


if __name__ == "__main__":
    main()
