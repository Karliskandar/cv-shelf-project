from pathlib import Path
from collections import Counter
import argparse


def count_split(split_path: Path) -> dict:
    """
    Count how many images and labels exist in one dataset split.
    Example split names: train, valid, val, test.
    """
    images_dir = split_path / "images"
    labels_dir = split_path / "labels"
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    images = (
        [p for p in images_dir.rglob("*") if p.suffix.lower() in image_extensions]
        if images_dir.exists()
        else []
    )
    labels = list(labels_dir.rglob("*.txt")) if labels_dir.exists() else []

    return {
        "images": len(images),
        "labels": len(labels),
    }


def parse_label_file(label_path: Path) -> list[int]:
    """
    Read one YOLO label file and return the class IDs found inside it.

    YOLO label format:
        class_id x_center y_center width height
    """
    class_ids = []

    if not label_path.exists():
        return class_ids

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            try:
                class_id = int(parts[0])
                class_ids.append(class_id)
            except ValueError:
                print(f"[WARN] Malformed line in {label_path}: {line.strip()!r}")
                continue

    return class_ids


def count_classes(dataset_root: Path, splits: list[str]) -> Counter:
    """
    Go through all label files in all splits and count how many times
    each class ID appears.
    """
    class_counter = Counter()

    for split in splits:
        labels_dir = dataset_root / split / "labels"
        if not labels_dir.exists():
            continue

        for label_file in labels_dir.rglob("*.txt"):
            class_ids = parse_label_file(label_file)
            class_counter.update(class_ids)

    return class_counter


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a YOLO-format dataset.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to dataset root containing train/valid/val/test folders.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)

    candidate_splits = ["train", "valid", "val", "test"]
    splits = [split for split in candidate_splits if (dataset_root / split).exists()]

    print("\n=== DATASET INSPECTION ===")
    print(f"Dataset root: {dataset_root.resolve()}")

    if not splits:
        print("\nNo dataset splits found.")
        print("Expected folders such as train/, valid/ or val/, and test/.")
        return

    for split in candidate_splits:
        split_path = dataset_root / split
        print(f"\n[{split.upper()}]")
        if split_path.exists():
            stats = count_split(split_path)
            print(f"  Images: {stats['images']}")
            print(f"  Labels: {stats['labels']}")
        else:
            print("  Split not found.")

    class_counts = count_classes(dataset_root, splits)

    print("\n=== CLASS DISTRIBUTION (BY CLASS ID) ===")
    if not class_counts:
        print("No class labels found.")
    else:
        for class_id, count in sorted(class_counts.items()):
            print(f"  Class {class_id}: {count} objects")


if __name__ == "__main__":
    main()