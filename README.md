# Automated Visual Quality Control Inspection (MVTec AD — Bottle)

Modular classical computer vision pipeline for automated industrial defect detection on overhead glass bottle images from the **MVTec Anomaly Detection (MVTec AD)** dataset. Built with Python, OpenCV, NumPy, and `pytest` using an iterative Agile workflow.

---

## 1. Pipeline Architecture

```text
data/golden_set/ ──► acquisition.py ──► preprocessing.py ──► inspection.py ──► postprocessing.py
                                              │                     ▲                  │
                                              ▼                     │                  ▼
                                    Centroid Alignment      Statistical Model    Annotated Images
                                    (cv2.moments + Warp)   (mean ± k*std .npz)   + outputs/report.csv
```

* **`configs/default.yaml`**: Centralizes directory paths, preprocessing parameters, and adaptive statistical thresholds.
* **`src/acquisition.py`**: Loads input images deterministically from target directories.
* **`src/preprocessing.py`**: Applies grayscale conversion, Gaussian smoothing, inverse binary thresholding, and centroid-based affine translation (`cv2.moments` + `cv2.warpAffine`) to align every bottle to the frame center.
* **`src/inspection.py`**: Compares each aligned sample against a pixel-wise statistical golden model ($\mu \pm (k\sigma + \text{offset})$) learned from 200 normal training samples, filtering noise via morphological opening and contour area thresholding.
* **`src/postprocessing.py`**: Draws defect bounding boxes and `PASS` / `FAIL` status overlays onto output images.
* **`src/pipeline.py`**: Orchestrates end-to-end batch inspection, caches the statistical reference model (`outputs/reference_model.npz`), and exports a structured audit log (`outputs/report.csv`).
* **`src/evaluate.py`**: Benchmarks classification performance (`Accuracy`, `Precision`, `Recall`, `F1-Score`, and Confusion Matrix) across all 83 MVTec test samples (`broken_large`, `broken_small`, `contamination`, `good`).

---

## 2. Quickstart

### Setup Environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run Automated Tests
```powershell
pytest
```

### Run Batch Inspection (Golden Set + CSV Report)
```powershell
python main.py
```
Outputs annotated images and `outputs/report.csv`.

### Run Full MVTec Test-Set Benchmark (83 Images)
*(Requires full MVTec `bottle` dataset extracted under `data/raw/`)*
```powershell
python -m src.evaluate
```

---

## 3. Iterative Engineering Evolution & Benchmark Results

1. **Global Thresholding (Sprint 1):** Failed by flagging dark amber glass against the white background.
2. **Canny Edge Detection + Eroded ROI:** Failed due to false positives on normal concentric bottle neck rings and specular reflections.
3. **Single-Reference Subtraction:** Failed due to rotational variance of bottle screw threads.
4. **Centroid-Aligned Statistical Reference Model (Sprint 2 & 3):** Solved rotational variance by modeling pixel-wise mean ($\mu$) and standard deviation ($\sigma$) across 200 aligned normal training bottles.

### Hyperparameter Operating Points (83 Test Samples)

| Operating Mode | `std_multiplier` | `min_defect_area_px` | `broken_large` | `broken_small` | `contamination` | `good` | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Conservative (Zero False Alarms)** | `3.5` | `250` | 20/20 (100%) | 21/22 (95.5%) | 16/21 (76.2%) | 20/20 (100%) | 92.77% | **100.00%** | 90.48% | **95.00%** |
| **Safety-First (Active Default)** | `2.8` | `150` | **20/20 (100%)** | **22/22 (100%)** | **19/21 (90.5%)** | 15/20 (75.0%) | 91.57% | 92.42% | **96.83%** | 94.57% |

*Default configuration prioritizes manufacturing safety (`96.83%` Recall, `100%` structural glass crack detection), reducing escaped defects from 6 down to 2.*

### Confusion Matrix (Active Default: `std_multiplier: 2.8`, `min_defect_area_px: 150`)

| Ground Truth \ Prediction | Predicted `FAIL` (Defective) | Predicted `PASS` (Good) | Total |
| :--- | :---: | :---: | :---: |
| **Actual Defective** (`broken_large`, `broken_small`, `contamination`) | **61** (True Positive) | **2** (False Negative) | 63 |
| **Actual Good** (`good`) | **5** (False Positive) | **15** (True Negative) | 20 |
| **Total** | 66 | 17 | **83** |