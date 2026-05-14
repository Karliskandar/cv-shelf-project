# ShelfVision — YOLO-Based Supermarket Shelf Product Detection and Out-of-Stock Analysis

Computer Vision final project: an end-to-end pipeline that detects supermarket shelf products using fine-tuned YOLOv8, counts visible product facings, and identifies low-stock categories.

## Classes

| ID | Class  | Expected Count (P75) |
|----|--------|---------------------|
| 0  | water  | 3                   |
| 1  | milk   | 2                   |
| 2  | juice  | 3                   |
| 3  | cereal | 2                   |
| 4  | chips  | 2                   |
| 5  | pasta  | 2                   |

## Dataset

**Groceries6** — 1,125 images remapped from a 25-class grocery dataset down to 6 target categories.

| Split | Images |
|-------|--------|
| Train | 815    |
| Val   | 205    |
| Test  | 105    |

The dataset is not included in this repository due to size. Place it at `data/processed/groceries6/` with the standard YOLO directory structure (`train/images`, `train/labels`, `valid/images`, `valid/labels`, `test/images`, `test/labels`).

## Setup

```bash
# Clone the repository
git clone https://github.com/Karliskandar/cv-shelf-project.git
cd cv-shelf-project

# Install dependencies
pip install -r requirements.txt

# Verify dataset path
# Open configs/groceries6.yaml and ensure the `path:` line points to your data location.
# The default relative path works if you run commands from the repo root:
#   path: data/processed/groceries6
```

## Training

Both models were trained with identical hyperparameters for a fair comparison:

```bash
# YOLOv8n baseline
yolo train model=yolov8n.pt data=configs/groceries6.yaml epochs=40 imgsz=640 batch=16 seed=42 project=runs/train name=groceries6_yolov8n_baseline

# YOLOv8s
yolo train model=yolov8s.pt data=configs/groceries6.yaml epochs=40 imgsz=640 batch=16 seed=42 project=runs/train name=yolov8s_groceries6
```

Training takes ~12 minutes on GPU (RTX 4070) or ~3 hours on CPU. Trained weights are saved to `runs/train/*/weights/best.pt`.

## Running Inference

```bash
# Step 1: Compute expected stock counts from training labels
python scripts/compute_expected_counts.py

# Step 2: Run detection on test images + stock analysis
python scripts/predict_and_analyze.py

# Step 3: Generate stock summary report and chart
python scripts/stock_summary.py
```

Output is saved to `runs/analysis/`:
- `annotated/` — test images with bounding boxes drawn
- `stock_report.csv` — per-image, per-class detection counts and stock status
- `stock_summary_chart.png` — visual summary of stock-status distribution

To run on a single image:

```bash
yolo predict model=runs/train/yolov8s_groceries6/weights/best.pt source=path/to/image.jpg conf=0.40
```

## Model Comparison

```bash
python scripts/compare_models.py
```

Generates `runs/comparison_n_vs_s.png` comparing YOLOv8n and YOLOv8s per-class mAP.

## Project Structure

```
cv-shelf-project/
├── configs/
│   ├── groceries6.yaml          # Dataset config (class names, paths)
│   └── stock_thresholds.json    # Expected counts + low-stock rules
├── data/
│   └── processed/groceries6/    # Dataset (gitignored)
├── runs/
│   ├── train/                   # Training outputs (gitignored)
│   ├── val/                     # Validation outputs (gitignored)
│   └── analysis/                # Stock analysis outputs (gitignored)
├── scripts/
│   ├── inspect_dataset.py       # Dataset statistics and validation
│   ├── remap_labels.py          # 25-class → 6-class label remapping
│   ├── compute_expected_counts.py  # P75 expected count computation
│   ├── predict_and_analyze.py   # Inference + stock status pipeline
│   ├── stock_summary.py         # Aggregate reporting
│   └── compare_models.py        # YOLOv8n vs YOLOv8s comparison
├── requirements.txt
├── HANDOFF.md                   # Detailed project handoff documentation
└── README.md
```

## Weights

Pre-trained base weights (`yolov8n.pt`, `yolov8s.pt`) are auto-downloaded by Ultralytics on first training run. Fine-tuned weights are saved to `runs/train/*/weights/best.pt` and excluded from git due to size. To reproduce, run the training commands above.

## Results

| Metric     | YOLOv8n | YOLOv8s |
|------------|---------|---------|
| Precision  | 0.818   | 0.833   |
| Recall     | 0.801   | 0.814   |
| mAP@50     | 0.862   | 0.881   |
| mAP@50-95  | 0.677   | 0.694   |
| Count MAE  | —       | 0.730   |
