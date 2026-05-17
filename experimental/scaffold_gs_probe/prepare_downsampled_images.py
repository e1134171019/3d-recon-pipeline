from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Build images_<factor> for gsplat scaffold datasets.")
    parser.add_argument("--src", type=Path, required=True, help="Source images directory")
    parser.add_argument("--factor", type=int, required=True, help="Downsample factor, e.g. 8")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality for output images")
    parser.add_argument("--dst", type=Path, help="Optional explicit output directory")
    args = parser.parse_args()

    if args.factor < 1:
        raise ValueError("factor must be >= 1")
    if not args.src.exists():
        raise FileNotFoundError(f"Source directory missing: {args.src}")

    dst = args.dst or args.src.parent / f"{args.src.name}_{args.factor}"
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in args.src.iterdir() if p.is_file())
    total = len(files)
    if total == 0:
        raise ValueError(f"No files found in {args.src}")

    for idx, src_path in enumerate(files, start=1):
        image = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {src_path}")
        height, width = image.shape[:2]
        out_w = max(width // args.factor, 1)
        out_h = max(height // args.factor, 1)
        resized = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)
        out_path = dst / src_path.name
        ok = cv2.imwrite(str(out_path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
        if not ok:
            raise ValueError(f"Failed to write image: {out_path}")
        if idx % 50 == 0 or idx == total:
            print(f"[{idx}/{total}] {out_path.name}")

    print(f"[DONE] Wrote {total} images to {dst}")


if __name__ == "__main__":
    main()
