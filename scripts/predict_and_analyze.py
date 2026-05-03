"""
predict_and_analyze.py  (v2 — only analyze detected products)

End-to-end shelf analysis pipeline:
  1. Loads the trained YOLOv8s model (best.pt).
  2. Runs detection on every image in data/processed/groceries6/test/images.
  3. For each image, identifies WHICH product classes are present (>=1 detection).
  4. Counts detections only for those present classes.
  5. Compares counts against expected thresholds (from configs/stock_thresholds.json).
  6. Classifies each (image, present-class) pair as OK / LOW_STOCK.

Why only detected classes?
  The Groceries dataset contains product-specific shelves — a cereal-shelf image
  typically has 0 milk, 0 chips, etc. Asking "is the milk shelf stocked?" on a
  cereal-shelf image is meaningless and would falsely flag every shelf as
  OUT_OF_STOCK for products that simply don't belong there.

  In a real deployment, an OUT_OF_STOCK detection would require external knowledge
  about which products SHOULD be on each shelf (e.g. a planogram / shelf map).
  We don't have that info, so we restrict our analysis to products actually
  present.

Outputs:
  - runs/analysis/annotated/        annotated images, one per test image
  - runs/analysis/stock_report.csv  per-image, per-PRESENT-class stock status

Usage (from repo root):
    python scripts/predict_and_analyze.py
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO


# ---------- Configuration ----------

MODEL_PATH = Path("runs/train/yolov8s_groceries6/weights/best.pt")
TEST_IMAGES_DIR = Path("data/processed/groceries6/test/images")
THRESHOLDS_CONFIG = Path("configs/stock_thresholds.json")

OUTPUT_DIR = Path("runs/analysis")
ANNOTATED_DIR = OUTPUT_DIR / "annotated"
CSV_REPORT = OUTPUT_DIR / "stock_report.csv"

# Detection confidence threshold. Higher = fewer false positives.
CONF_THRESHOLD = 0.40

CLASS_NAMES = ["water", "milk", "juice", "cereal", "chips", "pasta"]

CLASS_COLORS = {
    "water":  (255, 178, 102),
    "milk":   (255, 255, 255),
    "juice":  (0, 165, 255),
    "cereal": (0, 215, 255),
    "chips":  (0, 0, 255),
    "pasta":  (147, 20, 255),
}

STATUS_COLORS = {
    "OK":        (0, 200, 0),
    "LOW_STOCK": (0, 165, 255),
}


# ---------- Stock-status logic ----------

def classify_stock(count: int, expected: int, low_pct: float) -> str:
    """
    Two-tier classification (only called when count >= 1):
      - LOW_STOCK: count < (low_pct * expected)
      - OK:        count >= (low_pct * expected)
    """
    if count < low_pct * expected:
        return "LOW_STOCK"
    return "OK"


# ---------- Annotation helpers ----------

def draw_detection_boxes(image, detections):
    for class_name, x1, y1, x2, y2, conf in detections:
        color = CLASS_COLORS.get(class_name, (200, 200, 200))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return image


def draw_stock_status_panel(image, present_classes_status):
    """Draw a panel showing only classes detected in this image."""
    panel_x, panel_y = 10, 10
    line_height = 24
    panel_width = 300

    if not present_classes_status:
        panel_height = 52
    else:
        panel_height = line_height * (len(present_classes_status) + 1) + 14

    overlay = image.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_width, panel_y + panel_height),
                  (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)

    cv2.putText(image, "Stock Status:", (panel_x + 10, panel_y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    if not present_classes_status:
        cv2.putText(image, "(no products detected)",
                    (panel_x + 10, panel_y + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return image

    for i, (class_name, info) in enumerate(present_classes_status.items()):
        status = info["status"]
        text = f"{class_name}: {info['count']}/{info['expected']}  [{status}]"
        y = panel_y + 24 + (i + 1) * line_height
        cv2.putText(image, text, (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    STATUS_COLORS[status], 2)
    return image


# ---------- Main pipeline ----------

def main() -> None:
    if not THRESHOLDS_CONFIG.exists():
        raise SystemExit(
            f"{THRESHOLDS_CONFIG} not found. "
            f"Run scripts/compute_expected_counts.py first."
        )
    with open(THRESHOLDS_CONFIG, "r", encoding="utf-8") as f:
        thresholds = json.load(f)
    expected_counts = thresholds["expected_counts"]
    low_pct = thresholds["low_stock_threshold_pct"]

    print("Expected counts loaded:")
    for k, v in expected_counts.items():
        print(f"  {k}: {v}")
    print(f"Low-stock threshold: {low_pct * 100:.0f}% of expected\n")

    if not MODEL_PATH.exists():
        raise SystemExit(f"Model weights not found at {MODEL_PATH}")
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(str(MODEL_PATH))

    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in TEST_IMAGES_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    print(f"Found {len(image_paths)} test images.\n")

    results = model.predict(
        source=[str(p) for p in image_paths],
        conf=CONF_THRESHOLD,
        device=0,
        verbose=False,
        stream=False,
    )

    csv_rows = []
    summary_per_status = defaultdict(int)
    images_with_no_detections = 0

    for image_path, result in zip(image_paths, results):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  [WARN] Could not read {image_path.name}, skipping.")
            continue

        detections = []
        per_class_count = defaultdict(int)

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = CLASS_NAMES[class_id]
                conf = float(box.conf.item())
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())

                detections.append((class_name, x1, y1, x2, y2, conf))
                per_class_count[class_name] += 1

        # KEY CHANGE: only analyze classes that were detected (count >= 1).
        present_classes_status = {}
        for class_name in CLASS_NAMES:
            count = per_class_count.get(class_name, 0)
            if count == 0:
                continue  # Class not on this shelf — skip.
            expected = expected_counts[class_name]
            status = classify_stock(count, expected, low_pct)
            present_classes_status[class_name] = {
                "count": count,
                "expected": expected,
                "status": status,
            }
            summary_per_status[status] += 1

            csv_rows.append({
                "image": image_path.name,
                "class": class_name,
                "count": count,
                "expected": expected,
                "status": status,
            })

        if not present_classes_status:
            images_with_no_detections += 1

        image = draw_detection_boxes(image, detections)
        image = draw_stock_status_panel(image, present_classes_status)

        out_path = ANNOTATED_DIR / image_path.name
        cv2.imwrite(str(out_path), image)

    with open(CSV_REPORT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "class", "count", "expected", "status"],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Done. Annotated images saved to: {ANNOTATED_DIR.resolve()}")
    print(f"Stock report saved to:           {CSV_REPORT.resolve()}\n")
    print(f"Total observations: {len(csv_rows)}")
    print(f"Images with no detections: {images_with_no_detections}\n")
    print("Stock-status distribution across detected (image, class) pairs:")
    total = sum(summary_per_status.values())
    for status in ["OK", "LOW_STOCK"]:
        n = summary_per_status[status]
        pct = 100 * n / total if total else 0
        print(f"  {status:<14} {n:>5} ({pct:>5.1f}%)")


if __name__ == "__main__":
    main()
