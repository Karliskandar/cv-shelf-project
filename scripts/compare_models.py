"""
Compare YOLOv8n vs YOLOv8s on the test split.
Produces a bar chart and a text summary.

Both models were validated on the test split using `yolo mode=val split=test`.
This script just hardcodes the resulting numbers — we already have them from
the console output, no need to re-run validation here.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Test-split results from the validation runs
classes = ["water", "milk", "juice", "cereal", "chips", "pasta"]

# mAP50 per class
yolov8n_map50 = [0.976, 0.892, 0.875, 0.875, 0.826, 0.725]
yolov8s_map50 = [0.964, 0.945, 0.852, 0.865, 0.871, 0.786]

# mAP50-95 per class
yolov8n_map5095 = [0.739, 0.696, 0.735, 0.718, 0.662, 0.511]
yolov8s_map5095 = [0.782, 0.734, 0.737, 0.736, 0.654, 0.523]

# Plot setup
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(classes))
width = 0.35

# mAP50 chart
ax = axes[0]
ax.bar(x - width/2, yolov8n_map50, width, label="YOLOv8n", color="#4C72B0")
ax.bar(x + width/2, yolov8s_map50, width, label="YOLOv8s", color="#DD8452")
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylabel("mAP50")
ax.set_title("Per-class mAP50 (test split)")
ax.set_ylim(0.5, 1.0)
ax.legend()
ax.grid(axis="y", alpha=0.3)

# mAP50-95 chart
ax = axes[1]
ax.bar(x - width/2, yolov8n_map5095, width, label="YOLOv8n", color="#4C72B0")
ax.bar(x + width/2, yolov8s_map5095, width, label="YOLOv8s", color="#DD8452")
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylabel("mAP50-95")
ax.set_title("Per-class mAP50-95 (test split)")
ax.set_ylim(0.4, 0.85)
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

# Save
output_path = Path("runs") / "comparison_n_vs_s.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved chart to {output_path.resolve()}")

# Also print the summary table
print("\n=== Overall metrics (test split) ===")
print(f"{'Metric':<15} {'YOLOv8n':>10} {'YOLOv8s':>10} {'Δ':>10}")
print("-" * 47)
for name, n, s in [
    ("Precision",  0.818, 0.833),
    ("Recall",     0.801, 0.814),
    ("mAP50",      0.862, 0.881),
    ("mAP50-95",   0.677, 0.694),
]:
    delta = s - n
    print(f"{name:<15} {n:>10.3f} {s:>10.3f} {delta:>+10.3f}")