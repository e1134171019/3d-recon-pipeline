from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/annotations/bootstrap_12/dataset.yaml"),
    )
    parser.add_argument("--model", default="weights/yolo11n-seg.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="outputs/experiments/l0_semantic_roi")
    parser.add_argument("--name", default="bootstrap12_punch_holders_smoke")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        patience=20,
        cache=False,
        pretrained=True,
        workers=0,
    )


if __name__ == "__main__":
    main()
