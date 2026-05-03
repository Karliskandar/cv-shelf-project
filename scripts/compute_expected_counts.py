"""
compute_expected_counts.py  (v2 — 75th percentile)

Analyzes the training-set YOLO label files to compute per-class "expected" stock
counts. Writes the result to:

    configs/stock_thresholds.json

These thresholds answer the question:
    "How many objects of class X do we typically see on a fully-stocked shelf?"

Why we use the 75th PERCENTILE (instead of median):
- Median represents the *typical* shelf, which often has fewer items than a
  well-stocked one. Using median as "expected" makes LOW_STOCK structurally
  impossible to trigger because the LOW threshold (50% of expected) becomes 0
  or 1 for most classes.
- Maximum represents the BEST-stocked shelf in the dataset, often an outlier
  (e.g. a promotional display). Using max as "expected" inflates expectations.
- The 75th percentile splits the difference: it represents an UPPER-TYPICAL
  shelf — what a well-stocked, normally-operating store would display. This
  gives us a realistic baseline that produces meaningful LOW_STOCK rates.

The output JSON is human-editable. A real user/store manager can override
any value to match their actual store layout.

Usage (from repo root):
    python scripts/compute_expected_counts.py
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

# ---------- Configuration ----------

# Match the order in configs/groceries6.yaml
CLASS_NAMES = ["water", "milk", "juice", "cereal", "chips", "pasta"]

# Paths (relative to repo root)
TRAIN_LABELS_DIR = Path("data/processed/groceries6/train/labels")
OUTPUT_CONFIG = Path("configs/stock_thresholds.json")

# The percentage threshold below which stock is considered LOW.
LOW_STOCK_THRESHOLD_PCT = 0.50

# The percentile of per-image instance counts to use as "expected".
# 75 = upper-typical shelf (well-stocked but not exceptional).
EXPECTED_PERCENTILE = 75


def count_instances_per_image(labels_dir: Path) -> dict[int, list[int]]:
    """
    Walk every label file in `labels_dir`. For each file, count how many
    instances of each class are present. Return a dict mapping:
        class_id -> list of per-image counts (one entry per image where class appears)
    """
    per_class_counts: dict[int, list[int]] = defaultdict(list)

    label_files = list(labels_dir.glob("*.txt"))
    print(f"Found {len(label_files)} label files in {labels_dir}")

    for label_file in label_files:
        image_counts: dict[int, int] = defaultdict(int)

        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    class_id = int(parts[0])
                except ValueError:
                    continue
                image_counts[class_id] += 1

        for class_id, count in image_counts.items():
            per_class_counts[class_id].append(count)

    return per_class_counts


def compute_expected_count(counts: list[int], percentile: int) -> int:
    """
    Given a list of per-image counts for a class, return the expected count
    using the specified percentile, rounded to the nearest integer (min 1).
    """
    if not counts:
        return 1
    # statistics.quantiles splits into N equal groups -> N-1 cut points.
    # For the 75th percentile, we use n=4 and take the 3rd cut (index 2).
    if len(counts) == 1:
        return max(1, counts[0])
    quartiles = statistics.quantiles(counts, n=4)
    p75 = quartiles[2]  # third cut = 75th percentile
    return max(1, int(round(p75)))


def main() -> None:
    if not TRAIN_LABELS_DIR.exists():
        raise SystemExit(
            f"Training labels not found at {TRAIN_LABELS_DIR}. "
            f"Run from the repo root."
        )

    print(f"Computing expected stock counts using {EXPECTED_PERCENTILE}th "
          f"percentile of per-image instance counts...\n")
    per_class_counts = count_instances_per_image(TRAIN_LABELS_DIR)

    config = {
        "low_stock_threshold_pct": LOW_STOCK_THRESHOLD_PCT,
        "expected_count_method": f"p{EXPECTED_PERCENTILE}",
        "expected_counts": {},
        "_statistics": {},  # for transparency / defense
    }

    print(f"{'Class':<10} {'Images':>8} {'Min':>5} {'Median':>7} "
          f"{'P75':>5} {'Max':>5} {'Total':>7}")
    print("-" * 56)

    for class_id, class_name in enumerate(CLASS_NAMES):
        counts = per_class_counts.get(class_id, [])
        expected = compute_expected_count(counts, EXPECTED_PERCENTILE)

        if counts:
            try:
                p75 = statistics.quantiles(counts, n=4)[2] if len(counts) > 1 else counts[0]
            except statistics.StatisticsError:
                p75 = counts[0]
            stats = {
                "n_images": len(counts),
                "min": min(counts),
                "median": statistics.median(counts),
                "p75": round(p75, 2),
                "max": max(counts),
                "total_instances": sum(counts),
            }
        else:
            stats = {
                "n_images": 0,
                "min": 0,
                "median": 0,
                "p75": 0,
                "max": 0,
                "total_instances": 0,
            }

        config["expected_counts"][class_name] = expected
        config["_statistics"][class_name] = stats

        print(
            f"{class_name:<10} {stats['n_images']:>8} {stats['min']:>5} "
            f"{stats['median']:>7.1f} {stats['p75']:>5.1f} "
            f"{stats['max']:>5} {stats['total_instances']:>7}"
        )

    OUTPUT_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CONFIG, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\nExpected counts written to {OUTPUT_CONFIG.resolve()}")
    print("\nFinal expected_counts (used by predict_and_analyze.py):")
    for class_name, count in config["expected_counts"].items():
        print(f"  {class_name:<10} expected = {count}")
    print(
        f"\nLow-stock threshold: {LOW_STOCK_THRESHOLD_PCT * 100:.0f}% of expected"
    )
    print("(Edit the JSON manually to override any value for a specific store.)")


if __name__ == "__main__":
    main()
