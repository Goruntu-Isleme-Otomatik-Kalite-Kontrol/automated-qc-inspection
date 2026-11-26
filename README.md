<div align="center">

# 🔍 Industrial Automated Visual Quality Control System

**Real-Time Surface Defect Detection & Industrial HMI Platform Combining Classical Statistical Computer Vision (CIE LAB) and Hybrid Deep Learning (Patch Memory Bank)**

**`Domain:`** Industrial Computer Vision &nbsp;|&nbsp; **`Dataset:`** MVTec Anomaly Detection &nbsp;|&nbsp; **`Architecture:`** CIE LAB + ResNet-18 &nbsp;|&nbsp; **`Interface:`** Streamlit HMI

</div>

---

## 📌 Overview

An end-to-end visual quality control platform engineered for manufacturing lines where only **defect-free (*good*)** reference samples are available during calibration and defect morphologies are unknown in advance (*unsupervised anomaly detection*).

The system unifies two distinct computer vision paradigms under a single preprocessing, evaluation, and diagnostic pipeline:

| Paradigm | Core Methodology | Primary Advantage | Optimal Use Case |
| :--- | :--- | :--- | :--- |
| **Classical Statistical CV (`_cv`)** | Pixel-wise statistical modeling ($\mu \pm k\sigma$) in the CIE LAB color space with morphological filtering | Ultra-low latency (~61 FPS) and zero neural network overhead | Uniform textures under controlled industrial lighting |
| **Hybrid Deep Learning (`_nn`)** | OpenCV geometric normalization + multi-scale ResNet-18 L2-normalized Patch Memory Bank | High generalization across specular metallic reflections and complex textures | High-variance surfaces and subtle structural anomalies |

---

## 🚀 Release Milestones & Scope (`v1.0.0` → `v2.0.0`)

The platform was developed iteratively across three distinct architectural milestones:

| Version | Release Stage | Scope & Key Deliverables |
| :--- | :--- | :--- |
| **`v1.0.0`** | **Grayscale Statistical Baseline** | • Single-channel pixel mean and standard deviation ($\mu \pm k\sigma$) reference modeling.<br>• Otsu automatic background segmentation and morphological noise suppression (`cv2.MORPH_OPEN` / `MORPH_ELLIPSE`).<br>• Core `pytest` unit test suite and CLI evaluation baseline. |
| **`v1.1.0`** | **Multi-Channel CIE LAB Color Modeling** | • Migration from single-channel grayscale to 3-channel **CIE LAB** color space to decouple luminance (L<sup>*</sup>) fluctuations from chromatic (a<sup>*</sup>, b<sup>*</sup>) surface defects.<br>• Modular YAML configuration architecture (`configs/*.yaml`) per product category.<br>• Full-dataset evaluation pipeline generating F1-score, IoU, and throughput metrics. |
| **`v2.0.0`** | **Hybrid ResNet-18 & Industrial HMI** | • Integration of pre-trained **ResNet-18** (`layer2` + `layer3`) spatial patch feature extraction with an L2-normalized **Patch Memory Bank**, boosting `metal_nut` F1-score by **+29.28%**.<br>• Inverse affine coordinate alignment (`-dx, -dy`) mapping centered predictions back to raw image pixel coordinates.<br>• Multi-criteria Out-of-Distribution (**OOD / `INVALID`**) guard rejecting non-target objects.<br>• Dark-themed **Streamlit HMI** with side-by-side method comparison, live Ground Truth IoU/Dice verification, and `.png`/`.json` traceability export. |

---

## 🏗️ System Architecture

```text
                      ┌──────────────────────────────────┐
                      │      Raw Input Image (BGR)       │
                      └────────────────┬─────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 1. Preprocessing & Geometric Normalization (src/preprocessing.py)            │
│    • Gaussian Blur sensor noise suppression                                  │
│    • Otsu thresholding & corner pixel analysis for foreground segmentation   │
│    • Image moments (cv2.moments) centroid alignment shift (dx, dy)           │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                  ┌────────────────────┴────────────────────┐
                  ▼                                         ▼
┌───────────────────────────────────┐     ┌────────────────────────────────────┐
│ Classical Statistical Track (_cv) │     │ Hybrid Deep Learning Track (_nn)   │
│ • CIE LAB Reference Model (μ, σ)  │     │ • ResNet-18 Mid-Layer Features     │
│ • Pixel Difference Map |I - μ|    │     │ • L2-Normalized Patch Memory Bank  │
│ • Adaptive Thresholding (> k·σ)   │     │ • Nearest-Neighbor Euclidean Dist. │
└─────────────────┬─────────────────┘     └─────────────────┬──────────────────┘
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 2. Post-Processing, Inverse Alignment & OOD Guard (app.py / inspection.py)   │
│    • Morphological opening/closing & minimum contour area filtering          │
│    • Inverse affine transform (-dx, -dy) for pixel-exact box/heatmap overlay │
│    • Multi-criteria Out-of-Distribution (INVALID) rejection gate             │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
                 ┌────────────────────────────────────────────┐
                 │  Final Decision:  PASS  |  FAIL  | INVALID │
                 │  Anomaly Heatmap • Bounding Boxes • JSON   │
                 └────────────────────────────────────────────┘
```

<details>
<summary><b>📐 Mathematical Formulation & Decision Logic (Click to expand)</b></summary>
<br>

* **Centroid Alignment & Inverse Coordinate Mapping:**
  During preprocessing, the foreground binary mask $M(x,y)$ is used to compute the object centroid $(c_x, c_y)$ and shift the object to the frame center ($\Delta x = \frac{W}{2} - c_x, \; \Delta y = \frac{H}{2} - c_y$). Post-inference, predicted bounding boxes and anomaly heatmaps are shifted back via an inverse affine transformation to align pixel-for-pixel with the raw input frame:
  $$(x_{\text{orig}}, y_{\text{orig}}) = (x_{\text{pred}} - \Delta x, \; y_{\text{pred}} - \Delta y)$$

* **Patch-Level Anomaly Scoring:**
  Each extracted spatial feature vector $\mathbf{f}_i$ is L2-normalized onto the unit hypersphere ($\hat{\mathbf{f}}_i = \mathbf{f}_i / \Vert{}\mathbf{f}_i\Vert{}_2$) and scored by its minimum Euclidean distance to the defect-free reference memory bank $\mathcal{M}$:
  $$s_i = \min_{\mathbf{m} \in \mathcal{M}} \Vert{}\hat{\mathbf{f}}_i - \mathbf{m}\Vert{}_2$$

* **Out-of-Distribution (`INVALID`) Protection:**
  To prevent unrelated or out-of-category images from triggering false `PASS` or `FAIL` decisions, four threshold-independent guardrails are enforced: peak anomaly score ($\max(S) > 0.98$), 90th-percentile heatmap distribution ($P_{90}(S) > 0.78$), Otsu foreground occupancy ($\notin [0.10, 0.80]$), and geometric aspect ratio deviation ($> 0.25$).

</details>

---

## 📊 6-Way Benchmark Results & Performance Analysis

Evaluations were conducted across three industrial categories from the **MVTec Anomaly Detection** dataset (`bottle`, `metal_nut`, `hazelnut`), calibrated strictly on defect-free training samples.

| Configuration | Paradigm | Category | F1-Score (%) | Latency (ms/image) | Throughput (FPS) |
| :--- | :---: | :--- | :---: | :---: | :---: |
| `bottle_cv` | Classical CIE LAB | Bottle (`bottle`) | 95.0%+ | ~16.2 ms | ~61.7 |
| **`bottle_nn`** | **Hybrid ResNet-18** | **Bottle (`bottle`)** | **98.4%+** | **~24.5 ms** | **~40.8** |
| `metal_nut_cv` | Classical CIE LAB | Metal Nut (`metal_nut`) | 64.43% | ~16.5 ms | ~60.6 |
| **`metal_nut_nn`** | **Hybrid ResNet-18** | **Metal Nut (`metal_nut`)** | **93.71%** | **~27.8 ms** | **~36.0** |
| `hazelnut_cv` | Classical CIE LAB | Hazelnut (`hazelnut`) | 89.8%+ | ~16.4 ms | ~61.0 |
| **`hazelnut_nn`** | **Hybrid ResNet-18** | **Hazelnut (`hazelnut`)** | **94.0%+** | **~26.9 ms** | **~37.2** |

### Key Engineering Takeaways
* **Substantial Gain on Specular Metallic Surfaces:** While classical pixel differencing performs reliably on uniform backgrounds (`bottle`), metallic reflections and rotational variance in `metal_nut` degrade classical performance to `64.43%` F1. The hybrid patch memory bank achieves **`93.71%` F1** (**+29.28% improvement**).
* **Real-Time Conveyor Throughput:** Running at **~25–28 ms per frame (36–41 FPS)**, the hybrid pipeline satisfies real-time inline inspection constraints without sacrificing localization precision.

---

## 🔬 Empirical Failure Analysis & Trade-offs

* **Sensitivity vs. False-Alarm Trade-off (`metal_nut / color / 004.png`):**
  A micro-scale paint blemish along the inner bore of the metal nut registered an anomaly peak of `0.767` (exceeding the `0.72` threshold) and localized accurately on the heatmap. However, because its connected pixel area fell below `min_defect_area_px` (`120 px`) and was suppressed by morphological opening, the final classification returned `PASS`. Maintaining a conservative threshold (`0.72`) and minimum area filter prevents specular metallic glare on intact nuts from triggering false positives, stabilizing overall accuracy at **90.43%** while leaving sub-threshold micro-stains at the sensitivity boundary.
* **Pixel-Exact Coordinate Alignment (`hazelnut / cut / 014.png`):**
  Centroid shifting during preprocessing initially introduced a `15–25 px` spatial offset between detected bounding boxes and raw defect locations. Incorporating `compute_centering_shift` and inverse `cv2.warpAffine` mapping eliminated this discrepancy across both tracks.

---

## 🖥️ Industrial HMI Interface

The Streamlit diagnostic workstation (`app.py`) provides an operator-ready inspection environment:

* **Live Inspection & Visual Diagnostics:** Inspect MVTec test samples by defect category or upload external images for instant anomaly scoring and heatmap generation.
* **Side-by-Side Paradigm Comparison:** Toggle simultaneous evaluation to compare Classical CV and Hybrid AI outputs on the same sample in a single view.
* **Ground Truth Overlay & Live IoU / Dice:** Automatically loads MVTec pixel masks, renders green defect contours, and computes real-time sample-level IoU and Dice scores.
* **Dynamic Threshold & Heatmap Alpha Controls:** Adjust decision sensitivity and heatmap overlay opacity (`0.0` – `1.0`) in real time.
* **Inspection Traceability Export:** Download annotated inspection images (`.png`) alongside timestamped audit logs (`.json`) containing bounding box coordinates, anomaly scores, and latency metrics.

---

## ⚡ Installation & Usage

### 1. Environment Setup
```bash
git clone [https://github.com/UygarHfz/Goruntu-Isleme-Otomatik-Kalite-Kontrol.git](https://github.com/UygarHfz/Goruntu-Isleme-Otomatik-Kalite-Kontrol.git)
cd Goruntu-Isleme-Otomatik-Kalite-Kontrol
pip install -r requirements.txt
```

### 2. Run Automated Verification Tests
```bash
pytest -v
```

### 3. Generate Reference Caches & 6-Way Benchmark Report
```bash
python -m src.evaluate --all
```

### 4. Launch the Industrial HMI Dashboard
```bash
streamlit run app.py
```

---

## 📂 Repository Structure

```text
├── .streamlit/
│   └── config.toml             # Industrial dark theme configuration
├── configs/
│   ├── bottle_cv.yaml          # Bottle — Classical CIE LAB configuration
│   ├── bottle_nn.yaml          # Bottle — Hybrid ResNet-18 configuration
│   ├── metal_nut_cv.yaml       # Metal Nut — Classical CIE LAB configuration
│   ├── metal_nut_nn.yaml       # Metal Nut — Hybrid ResNet-18 configuration
│   ├── hazelnut_cv.yaml        # Hazelnut — Classical CIE LAB configuration
│   └── hazelnut_nn.yaml        # Hazelnut — Hybrid ResNet-18 configuration
├── src/
│   ├── preprocessing.py        # Foreground segmentation & centroid alignment
│   ├── inspection.py           # Classical statistical defect detection
│   ├── nn_detector.py          # ResNet-18 patch memory bank inference engine
│   └── evaluate.py             # Multi-configuration benchmark & metrics pipeline
├── tests/                      # Automated unit & integration test suite
├── results/                    # Benchmark CSV summaries & reference caches
├── app.py                      # Streamlit HMI diagnostic application
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```
