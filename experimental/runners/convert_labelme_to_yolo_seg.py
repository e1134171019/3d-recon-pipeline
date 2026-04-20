from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml


def _shape_to_polygon(shape: dict) -> list[tuple[float, float]]:
    shape_type = shape.get("shape_type", "polygon")
    points = shape.get("points", [])
    if not points:
        return []

    if shape_type == "rectangle" and len(points) == 2:
        (x1, y1), (x2, y2) = points
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        return [
            (left, top),
            (right, top),
            (right, bottom),
            (left, bottom),
        ]

    return [(float(x), float(y)) for x, y in points]


def _normalize_polygon(
    polygon: list[tuple[float, float]], width: float, height: float
) -> list[tuple[float, float]]:
    if width <= 0 or height <= 0:
        return []
    normalized = []
    for x, y in polygon:
        nx = min(max(x / width, 0.0), 1.0)
        ny = min(max(y / height, 0.0), 1.0)
        normalized.append((nx, ny))
    return normalized


def _polygon_to_yolo_line(class_id: int, polygon: list[tuple[float, float]]) -> str:
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in polygon)
    return f"{class_id} {coords}"


def convert_labelme_dir(
    images_dir: Path,
    labels_dir: Path,
    dataset_yaml: Path,
    class_name: str,
    accepted_labels: set[str],
) -> dict[str, int]:
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(images_dir.glob("*.jpg"))
    label_counter: Counter[str] = Counter()
    positives = 0
    negatives = 0

    for image_path in image_files:
        json_path = image_path.with_suffix(".json")
        out_path = labels_dir / f"{image_path.stem}.txt"
        lines: list[str] = []

        if json_path.exists():
            data = json.loads(json_path.read_text(encoding="utf-8"))
            width = float(data.get("imageWidth", 0))
            height = float(data.get("imageHeight", 0))

            for shape in data.get("shapes", []):
                label = str(shape.get("label", "")).strip()
                label_counter[label] += 1
                if label not in accepted_labels:
                    continue
                polygon = _shape_to_polygon(shape)
                polygon = _normalize_polygon(polygon, width, height)
                if len(polygon) < 3:
                    continue
                lines.append(_polygon_to_yolo_line(0, polygon))

        out_path.write_text("\n".join(lines), encoding="utf-8")
        if lines:
            positives += 1
        else:
            negatives += 1

    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(images_dir.parent).replace("\\", "/"),
                "train": "images",
                "val": "images",
                "names": {0: class_name},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    kept_shapes = sum(label_counter[label] for label in accepted_labels if label in label_counter)
    return {
        "images": len(image_files),
        "positives": positives,
        "negatives": negatives,
        "kept_shapes": kept_shapes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/annotations/bootstrap_12/images"),
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("data/annotations/bootstrap_12/labels"),
    )
    parser.add_argument(
        "--dataset-yaml",
        type=Path,
        default=Path("data/annotations/bootstrap_12/dataset.yaml"),
    )
    parser.add_argument("--class-name", default="punch_holders")
    parser.add_argument(
        "--accepted-label",
        action="append",
        dest="accepted_labels",
        default=["punch_holders", "上夾模塊"],
    )
    args = parser.parse_args()

    accepted_labels = {label.strip() for label in args.accepted_labels if label.strip()}
    summary = convert_labelme_dir(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        dataset_yaml=args.dataset_yaml,
        class_name=args.class_name,
        accepted_labels=accepted_labels,
    )

    print("Converted Labelme -> YOLO-seg")
    print(f"images       : {summary['images']}")
    print(f"positives    : {summary['positives']}")
    print(f"negatives    : {summary['negatives']}")
    print(f"kept_shapes  : {summary['kept_shapes']}")
    print(f"class_name   : {args.class_name}")
    print(f"accepted     : {sorted(accepted_labels)}")
    print(f"dataset_yaml : {args.dataset_yaml.resolve()}")


if __name__ == "__main__":
    main()
