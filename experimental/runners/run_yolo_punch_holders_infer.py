from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--project", default="outputs/experiments/l0_semantic_roi")
    parser.add_argument("--name", default="punch_holders_infer")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    model.predict(
        source=str(args.source),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        save=True,
        save_txt=False,
        save_conf=False,
        show=False,
    )


if __name__ == "__main__":
    main()
