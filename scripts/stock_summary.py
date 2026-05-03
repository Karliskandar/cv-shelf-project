"""
stock_summary.py  (v2 — only summarizes detected products)

Reads runs/analysis/stock_report.csv and produces:
  1. A per-class summary table showing OK vs LOW_STOCK across detected images.
  2. A bar chart visualizing the distribution.

Note: rows with status OUT_OF_STOCK are no longer produced by the v2 pipeline.
This script handles them gracefully if encountered (e.g. from an older CSV).

Usage (from repo root):
    python scripts/stock_summary.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CSV_REPORT = Path("runs/analysis/stock_report.csv")
OUTPUT_CHART = Path("runs/analysis/stock_summary_chart.png")
CLASS_NAMES = ["water", "milk", "juice", "cereal", "chips", "pasta"]
STATUSES = ["OK", "LOW_STOCK", "OUT_OF_STOCK"]  # OUT_OF_STOCK kept for backward compat


def load_report(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise SystemExit(
            f"{csv_path} not found. Run scripts/predict_and_analyze.py first."
        )
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def aggregate_by_class(rows: list[dict]) -> dict[str, dict[str, int]]:
    counts = {cls: {status: 0 for status in STATUSES} for cls in CLASS_NAMES}
    for row in rows:
        cls = row["class"]
        status = row["status"]
        if cls in counts and status in STATUSES:
            counts[cls][status] += 1
    return counts


def print_summary_table(counts: dict[str, dict[str, int]]) -> None:
    print(f"\n{'Class':<10} {'Detected':>9} {'OK':>5} {'LOW':>5} {'LOW %':>7}")
    print("-" * 42)
    for cls in CLASS_NAMES:
        c = counts[cls]
        total_detected = c["OK"] + c["LOW_STOCK"]
        low_pct = (100 * c["LOW_STOCK"] / total_detected) if total_detected else 0
        print(
            f"{cls:<10} {total_detected:>9} {c['OK']:>5} "
            f"{c['LOW_STOCK']:>5} {low_pct:>6.1f}%"
        )
    print()


def plot_chart(counts: dict[str, dict[str, int]], output_path: Path) -> None:
    classes = CLASS_NAMES
    ok_counts = [counts[c]["OK"] for c in classes]
    low_counts = [counts[c]["LOW_STOCK"] for c in classes]

    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, ok_counts, label="OK", color="#2E8B57")
    ax.bar(x, low_counts, bottom=ok_counts, label="LOW_STOCK", color="#FFA500")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Number of test images where class was detected")
    ax.set_title("Stock-status distribution per class (test set, detected products only)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved to {output_path.resolve()}")


def main() -> None:
    rows = load_report(CSV_REPORT)
    print(f"Loaded {len(rows)} rows from {CSV_REPORT}")

    counts = aggregate_by_class(rows)
    print_summary_table(counts)
    plot_chart(counts, OUTPUT_CHART)

    worst_class = max(
        CLASS_NAMES,
        key=lambda c: counts[c]["LOW_STOCK"],
    )
    worst_low = counts[worst_class]["LOW_STOCK"]
    worst_total = counts[worst_class]["OK"] + counts[worst_class]["LOW_STOCK"]
    if worst_total > 0:
        print(
            f"Most LOW_STOCK class: {worst_class} "
            f"({worst_low}/{worst_total} detected images flagged as LOW)"
        )


if __name__ == "__main__":
    main()
