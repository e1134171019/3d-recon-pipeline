from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import yaml


def _copy_pair(image_src: Path, label_src: Path, image_dst: Path, label_dst: Path) -> None:
    image_dst.parent.mkdir(parents=True, exist_ok=True)
    label_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_src, image_dst)
    shutil.copy2(label_src, label_dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("data/annotations/bootstrap_12"))
    parser.add_argument("--output-root", type=Path, default=Path("data/annotations/bootstrap_12_split"))
    parser.add_argument("--class-name", default="punch_holders")
    parser.add_argument("--val-positive-count", type=int, default=2)
    parser.add_argument("--val-negative-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_images = args.source_root / "images"
    source_labels = args.source_root / "labels"
    output_root = args.output_root

    pairs: list[tuple[str, Path, Path]] = []
    positives: list[tuple[Path, Path]] = []
    negatives: list[tuple[Path, Path]] = []

    for image_path in sorted(source_images.glob("*.jpg")):
        label_path = source_labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        text = label_path.read_text(encoding="utf-8").strip()
        if text:
            positives.append((image_path, label_path))
        else:
            negatives.append((image_path, label_path))

    rnd = random.Random(args.seed)
    rnd.shuffle(positives)
    rnd.shuffle(negatives)

    val_positives = positives[: min(args.val_positive_count, len(positives))]
    val_negatives = negatives[: min(args.val_negative_count, len(negatives))]
    train_positives = positives[len(val_positives) :]
    train_negatives = negatives[len(val_negatives) :]

    split_map = {
        "train": train_positives + train_negatives,
        "val": val_positives + val_negatives,
    }

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for split_name, items in split_map.items():
        for image_path, label_path in items:
            _copy_pair(
                image_path,
                label_path,
                output_root / split_name / "images" / image_path.name,
                output_root / split_name / "labels" / label_path.name,
            )

    dataset_yaml = {
        "path": str(output_root.resolve()).replace("\\", "/"),
        "train": "train/images",
        "val": "val/images",
        "names": {0: args.class_name},
    }
    (output_root / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_yaml, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    print("Prepared YOLO seg split")
    print(f"source_root       : {args.source_root.resolve()}")
    print(f"output_root       : {output_root.resolve()}")
    print(f"train positives   : {len(train_positives)}")
    print(f"train negatives   : {len(train_negatives)}")
    print(f"val positives     : {len(val_positives)}")
    print(f"val negatives     : {len(val_negatives)}")
    print(f"dataset_yaml      : {(output_root / 'dataset.yaml').resolve()}")


if __name__ == "__main__":
    main()
