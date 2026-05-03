# Handoff Guide — Continuing The CV Shelf Project

This document is everything you need to pick up the project where I left off. Read it once, then follow the setup checklist at the end.

---

## What I Did While I Had The Project

1. **Verified the dataset** you sent me (`data/processed/groceries6/` — 815 train / 205 val / 105 test images).
2. **Re-validated your YOLOv8n model** on the test split for a fair n-vs-s comparison.
3. **Trained YOLOv8s** on GPU (RTX 4070 Laptop), 40 epochs, identical hyperparameters to your YOLOv8n baseline (epochs=40, imgsz=640, batch=16, seed=42).
4. **Compared the two models** on the held-out test set.
5. **Built three new analysis scripts** for stock monitoring:
   - `compute_expected_counts.py` — derives per-class expected stock from training data (75th percentile method).
   - `predict_and_analyze.py` — runs detection on test images, counts per class, flags LOW_STOCK.
   - `stock_summary.py` — aggregates results into a per-class report and chart.
6. **Wrote up defense materials** (`talking_points.md` and `defense_prep_guide.md`).

---

## Headline Results

**YOLOv8n vs YOLOv8s on test split (105 images, 235 instances):**

| Metric    | YOLOv8n | YOLOv8s | Δ      |
|-----------|---------|---------|--------|
| Precision | 0.818   | 0.833   | +0.015 |
| Recall    | 0.801   | 0.814   | +0.013 |
| mAP50     | 0.862   | 0.881   | +0.019 |
| mAP50-95  | 0.677   | 0.694   | +0.017 |

YOLOv8s wins on every metric, with the largest per-class gains on the historically-hardest classes (pasta +6.1 mAP50, milk +5.3, chips +4.5). Marginal differences on already-easy classes (water, juice).

---

## ⚠️ IMPORTANT — Path Configuration Issue

The `configs/groceries6.yaml` file uses an **absolute path** that points to where the dataset lives on a specific machine. Every time the project moves between machines, this path needs to be updated.

### What To Do When You Pull This

Open `configs/groceries6.yaml`. The current first line on my machine is:

```yaml
path: ../data/processed/groceries6
```

This is a **relative path** that should work on any machine, *provided*:
- You run all commands from the repo root (the folder containing `configs/`, `scripts/`, etc.)
- The dataset is at `data/processed/groceries6/` inside the repo.

If for any reason it doesn't resolve, change it to your machine's absolute path. On your original setup it was:

```yaml
path: C:\Users\Admin\OneDrive\Documents\Shit\year2\Semester2\Computer Vision\CV Final Project\cv-shelf-project\data\processed\groceries6
```

Either form works as long as it's correct for your machine.

### Note On The Auto-Generated YAML

If you re-run `scripts/remap_labels.py` for any reason, it auto-regenerates the YAML with `output_root.resolve()` — meaning it'll write *your* current machine's absolute path. That's fine, but if you later move the project, you'll have to update the YAML again or revert to the relative path above.

---

## Files That Are NOT In Git — You Already Have These

The repo's `.gitignore` excludes large or machine-specific files. You already have all of them locally because they came from your original setup:

| Item                                      | Location                                   | Where it came from                |
|-------------------------------------------|--------------------------------------------|-----------------------------------|
| Processed dataset                         | `data/processed/groceries6/`               | Your original work                |
| YOLOv8n training run                      | `runs/train/groceries6_yolov8n_baseline/`  | Your original training            |

You don't need anything from me to recreate these — they're already on your machine.

---

## Files I'm Sending Manually (NOT in Git)

Because training runs aren't pushed to GitHub, I'm sending you the YOLOv8s outputs separately so you can verify my work without retraining. See the accompanying zip file.

| Item                          | Where to put it                            | Why you might want it             |
|-------------------------------|--------------------------------------------|-----------------------------------|
| YOLOv8s training run folder   | `runs/train/yolov8s_groceries6/`           | The trained weights + curves      |
| YOLOv8n test validation       | `runs/val/yolov8n_groceries6_test/`        | n model evaluated on test split   |
| YOLOv8s test validation       | `runs/val/yolov8s_groceries6_test/`        | s model evaluated on test split   |
| Stock analysis output         | `runs/analysis/`                           | Annotated images + CSV report     |

If you want to skip restoring these and re-run everything from scratch, that's fine — every script is reproducible. Just be aware training takes ~12 minutes on GPU (or ~3 hours on CPU).

---

## Setup Checklist For Resuming Work

Run through these in order on your machine.

### Step 1 — Pull the latest code

```powershell
cd <wherever you keep cv-shelf-project>
git pull
```

### Step 2 — Verify the dataset is in place

You should already have it. Confirm:

```powershell
(Get-ChildItem data\processed\groceries6\train\images\).Count
# Should be 815
```

If it returns 0 or errors, you need to put your `data/processed/groceries6/` back where it was.

### Step 3 — Verify Python dependencies

```powershell
python -c "import ultralytics; print(ultralytics.__version__)"
# Should print 8.4.45 or similar
```

If it errors with "No module named 'ultralytics'":

```powershell
pip install ultralytics
```

### Step 4 — Check the YAML path

Open `configs/groceries6.yaml`. Run the verification command:

```powershell
python -c "from ultralytics.data.utils import check_det_dataset; info = check_det_dataset('configs/groceries6.yaml'); print('Train:', info['train']); print('Classes:', info['names'])"
```

If this prints valid paths and your 6 class names, the dataset wiring is correct.

If it errors with "dataset not found," edit `configs/groceries6.yaml` and update the `path:` line to your machine's actual location.

### Step 5 — (Optional) Restore my YOLOv8s training run

If you want to inspect my trained model without retraining:

1. Unzip the manually-sent file into the project root.
2. Verify with:

```powershell
Test-Path runs\train\yolov8s_groceries6\weights\best.pt
# Should print True
```

### Step 6 — (Optional) Run the stock analysis pipeline

If everything above works:

```powershell
python scripts/compute_expected_counts.py
python scripts/predict_and_analyze.py
python scripts/stock_summary.py
```

This regenerates `runs/analysis/` with annotated images and the CSV report.

---

## What's Next On The Project

A few directions, in order of likely usefulness:

1. **Build a demo for the defense.** Live runs are more compelling than screenshots. A 5-minute demo running `predict_and_analyze.py` on a single test image and showing the annotated output would be straightforward to set up.
2. **Multi-seed training runs.** Re-train both models with seeds 42, 123, 7 to report mean ± std on the comparison metrics. Makes the comparison more rigorous.
3. **Train YOLOv8m** if there's time. Probably won't beat YOLOv8s by much given the dataset size, but it's a reasonable extra data point for the report.
4. **Class-targeted augmentation for pasta.** Pasta is the weakest class for both models. Heavy MixUp or oversampling could close the gap.

The defense prep documents (`talking_points.md`, `defense_prep_guide.md`) cover everything we have so far in question-and-answer form. Read them before the defense.

---

## Quick Reference

| Command | What it does |
|---------|--------------|
| `python scripts/inspect_dataset.py` | Original dataset inspection (your script) |
| `python scripts/remap_labels.py` | Original 25→6 class remapping (your script) |
| `python scripts/compute_expected_counts.py` | Generates `configs/stock_thresholds.json` |
| `python scripts/predict_and_analyze.py` | Runs YOLOv8s on test set, produces stock report |
| `python scripts/stock_summary.py` | Summarizes the stock report |

Training and validation use the `yolo` CLI (or `& "$env:APPDATA\Python\Python310\Scripts\yolo.exe" ...` on Windows if `yolo` isn't on PATH). Standard hyperparameters: `epochs=40 imgsz=640 batch=16 device=0 workers=2 seed=42`.
