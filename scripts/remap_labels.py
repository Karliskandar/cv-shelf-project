from pathlib import Path
import shutil
import argparse
import yaml


TARGET_CLASSES = ["water", "milk", "juice", "cereal", "chips", "pasta"]


def load_yaml_classes(yaml_path: Path) -> list[str]:
    """
    Load class names from a YOLO dataset YAML file.
    Supports either:
    - names: [class1, class2, ...]
    - names: {0: class1, 1: class2, ...}
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names", [])

    if isinstance(names, list):
        return names

    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys())]

    raise ValueError("Unsupported 'names' format in data.yaml")


def build_class_mapping(
    original_names: list[str], target_names: list[str]
) -> tuple[dict[int, int], dict[int, str]]:
    """
    Build mapping from original class IDs to new class IDs.
    """
    original_name_to_id = {name: idx for idx, name in enumerate(original_names)}
    old_to_new = {}
    kept_old_to_name = {}

    for new_id, class_name in enumerate(target_names):
        if class_name not in original_name_to_id:
            raise ValueError(f"Target class '{class_name}' not found in original dataset.")

        old_id = original_name_to_id[class_name]
        old_to_new[old_id] = new_id
        kept_old_to_name[old_id] = class_name

    return old_to_new, kept_old_to_name


def copy_image_if_needed(src_image: Path, dst_image: Path) -> None:
    dst_image.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_image, dst_image)


def remap_label_file(
    src_label: Path, dst_label: Path, old_to_new: dict[int, int]
) -> tuple[bool, int]:
    """
    Keep only target classes from one YOLO label file and remap them.

    Returns:
    - whether at least one object was kept
    - number of kept objects
    """
    kept_lines = []

    with open(src_label, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if not parts:
                continue

            if len(parts) != 5:
                print(f"  [WARN] Malformed line in {src_label}: {line.strip()!r} — skipping")
                continue

            try:
                old_class_id = int(parts[0])
            except ValueError:
                print(f"  [WARN] Non-integer class ID in {src_label}: {parts[0]!r} — skipping")
                continue

            if old_class_id not in old_to_new:
                continue

            new_class_id = old_to_new[old_class_id]
            new_line = " ".join([str(new_class_id)] + parts[1:])
            kept_lines.append(new_line)

    if not kept_lines:
        return False, 0

    dst_label.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_label, "w", encoding="utf-8") as f:
        f.write("\n".join(kept_lines) + "\n")

    return True, len(kept_lines)


def process_split(
    dataset_root: Path, output_root: Path, split: str, old_to_new: dict[int, int]
) -> tuple[int, int]:
    """
    Process one split and return:
    - kept images count
    - kept objects count
    """
    src_images_dir = dataset_root / split / "images"
    src_labels_dir = dataset_root / split / "labels"

    dst_images_dir = output_root / split / "images"
    dst_labels_dir = output_root / split / "labels"

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    kept_images = 0
    kept_objects = 0

    # Guard: skip gracefully if images dir is missing
    if not src_images_dir.exists():
        print(f"  [WARN] images/ directory not found in '{split}' split, skipping.")
        return 0, 0
    
    if not src_labels_dir.exists():
        print(f"  [WARN] labels/ directory not found in '{split}' split, skipping." )
        return 0, 0

    for image_path in src_images_dir.rglob("*"):
        if image_path.suffix.lower() not in image_extensions:
            continue

        relative_path = image_path.relative_to(src_images_dir)
        label_path = src_labels_dir / relative_path.with_suffix(".txt")

        if not label_path.exists():
            continue

        dst_image_path = dst_images_dir / relative_path
        dst_label_path = dst_labels_dir / relative_path.with_suffix(".txt")

        kept, object_count = remap_label_file(label_path, dst_label_path, old_to_new)

        if kept:
            copy_image_if_needed(image_path, dst_image_path)
            kept_images += 1
            kept_objects += object_count

    return kept_images, kept_objects


def write_output_yaml(output_root: Path, target_names: list[str]) -> Path:
    """
    Write the training YAML for the processed 6-class dataset.
    Saved alongside the processed data at output_root/groceries6.yaml.
    """
    yaml_content = {
        "path": str(output_root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": target_names,
    }

    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)

    out_path = configs_dir / "groceries6.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False, allow_unicode=True)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Remap a YOLO dataset to selected target classes.")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to original dataset root.")
    parser.add_argument("--output-root", type=str, required=True, help="Path to processed dataset output root.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)

    yaml_path = dataset_root / "data.yaml"
    original_names = load_yaml_classes(yaml_path)

    print("\n=== ORIGINAL CLASSES ===")
    for idx, name in enumerate(original_names):
        print(f"  {idx}: {name}")

    old_to_new, kept_old_to_name = build_class_mapping(original_names, TARGET_CLASSES)

    print("\n=== CLASS REMAPPING ===")
    for old_id, new_id in sorted(old_to_new.items()):
        print(f"  {old_id} ({kept_old_to_name[old_id]}) -> {new_id} ({TARGET_CLASSES[new_id]})")

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print("\n=== PROCESSING SPLITS ===")
    for split in ["train", "valid", "test"]:
        split_path = dataset_root / split
        if not split_path.exists():
            print(f"  {split}: not found, skipping")
            continue

        kept_images, kept_objects = process_split(dataset_root, output_root, split, old_to_new)
        print(f"  {split}: kept {kept_images} images, {kept_objects} objects")

    config_path = write_output_yaml(output_root, TARGET_CLASSES)

    print("\nDone.")
    print(f"  Processed dataset saved to : {output_root.resolve()}")
    print(f"  Training config written to : {config_path.resolve()}")


if __name__ == "__main__":
    main()