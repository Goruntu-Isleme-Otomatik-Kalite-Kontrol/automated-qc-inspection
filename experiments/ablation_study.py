import csv
from pathlib import Path
import cv2
import numpy as np
from src.acquisition import load_images
from src.preprocessing import preprocess_image
from src.pipeline import get_statistical_reference

BOTTLE_TRAIN_DIR = "data/raw/bottle/train/good"
BOTTLE_TEST_DIR = Path("data/raw/bottle/test")
REFERENCE_CACHE = "outputs/reference_model.npz"
GOLDEN_SET_DIR = "data/golden_set"
FIGURE_DIR = Path("docs/images")
SUMMARY_CSV = Path("results/ablation_summary.csv")

KERNEL_SIZE = 9
BG_THRESHOLD = 210
# Same decision rule for every method: an image is FAIL if at least this many pixels are flagged
# (matches min_defect_area_px of the v1.0 baseline in configs/default.yaml)
MIN_FLAGGED_PX = 150

METHOD_NAMES = [
    "1. Global Threshold",
    "2. Canny + ROI",
    "3. Single Reference",
    "4. Mean + Std Model",
]


def method_1_global_threshold(gray: np.ndarray) -> np.ndarray:
    # Fails: flags the entire dark amber bottle body against the white background
    _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    return mask


def method_2_canny_roi(gray: np.ndarray) -> np.ndarray:
    # Fails: concentric bottle neck rings and specular reflections trigger false edges
    edges = cv2.Canny(gray, 50, 150)
    h, w = gray.shape
    roi = np.zeros_like(gray)
    cv2.circle(roi, (w // 2, h // 2), int(min(h, w) * 0.36), 255, -1)
    return cv2.bitwise_and(edges, roi)


def method_3_single_reference(aligned: np.ndarray, ref_aligned: np.ndarray) -> np.ndarray:
    # Fails: rotational mismatch of screw threads creates false crescent alarms
    diff = cv2.absdiff(aligned, ref_aligned)
    _, mask = cv2.threshold(diff, 45, 255, cv2.THRESH_BINARY)
    return mask


def method_4_statistical_model(
    aligned: np.ndarray, mean_img: np.ndarray, std_img: np.ndarray
) -> np.ndarray:
    # Succeeds: adaptive tolerance (2.8 * std + 25) suppresses rotating screw threads
    diff = np.abs(aligned.astype(np.float32) - mean_img)
    tolerance = (2.8 * std_img) + 25.0
    mask = (diff > tolerance).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def run_all_methods(
    aligned: np.ndarray, ref_aligned: np.ndarray, mean_img: np.ndarray, std_img: np.ndarray
) -> list[np.ndarray]:
    return [
        method_1_global_threshold(aligned),
        method_2_canny_roi(aligned),
        method_3_single_reference(aligned, ref_aligned),
        method_4_statistical_model(aligned, mean_img, std_img),
    ]


def save_visual_comparisons(
    golden: dict, ref_aligned: np.ndarray, mean_img: np.ndarray, std_img: np.ndarray
) -> None:
    """Qualitative check: render the four methods side by side on two golden-set samples."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    titles = [
        "1. Global Thresh (Fail)",
        "2. Canny + ROI (Fail)",
        "3. Single Ref (Fail)",
        "4. Mean+Std Model (Pass)",
    ]

    for sample_name in ["pass_02.png", "fail_01.png"]:
        aligned = preprocess_image(golden[sample_name], KERNEL_SIZE, BG_THRESHOLD)
        masks = run_all_methods(aligned, ref_aligned, mean_img, std_img)

        panels = []
        for title, mask in zip(titles, masks):
            vis = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
            vis[mask > 0] = (0, 0, 255)  # Highlight flagged defect pixels in red
            vis = cv2.resize(vis, (320, 320))
            cv2.putText(vis, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
            panels.append(vis)

        save_path = FIGURE_DIR / f"ablation_{sample_name}"
        cv2.imwrite(str(save_path), np.hstack(panels))
        print(f"Saved visual comparison: {save_path}")


def evaluate_on_test_set(
    ref_aligned: np.ndarray, mean_img: np.ndarray, std_img: np.ndarray
) -> list[dict]:
    """Quantitative check: run all four methods on every image of the bottle test set."""
    if not BOTTLE_TEST_DIR.exists():
        raise FileNotFoundError(
            f"Test set not found: {BOTTLE_TEST_DIR}. See README section 'Download the MVTec AD Dataset'."
        )

    stats = [{"tp": 0, "tn": 0, "fp": 0, "fn": 0, "good_area_pct": []} for _ in METHOD_NAMES]

    for category_dir in sorted(BOTTLE_TEST_DIR.iterdir()):
        if not category_dir.is_dir():
            continue
        is_defective = category_dir.name != "good"
        print(f"[ablation] Kategori işleniyor: {category_dir.name} ...", flush=True)

        for _, img in load_images(str(category_dir)):
            aligned = preprocess_image(img, KERNEL_SIZE, BG_THRESHOLD)
            masks = run_all_methods(aligned, ref_aligned, mean_img, std_img)

            for s, mask in zip(stats, masks):
                flagged_px = int(np.count_nonzero(mask))
                predicted_defect = flagged_px >= MIN_FLAGGED_PX

                if is_defective:
                    s["tp" if predicted_defect else "fn"] += 1
                else:
                    s["fp" if predicted_defect else "tn"] += 1
                    # How much of an intact bottle is wrongly flagged (false-alarm area)
                    s["good_area_pct"].append(100.0 * flagged_px / mask.size)

    rows = []
    for name, s in zip(METHOD_NAMES, stats):
        tp, tn, fp, fn = s["tp"], s["tn"], s["fp"], s["fn"]
        total = tp + tn + fp + fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "method": name,
                "accuracy_pct": round(100 * (tp + tn) / total, 2) if total else 0.0,
                "precision_pct": round(100 * precision, 2),
                "recall_pct": round(100 * recall, 2),
                "specificity_pct": round(100 * specificity, 2),
                "f1_pct": round(100 * f1, 2),
                "false_alarm_area_pct": round(float(np.mean(s["good_area_pct"])), 2)
                if s["good_area_pct"]
                else 0.0,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            }
        )
    return rows


def print_and_save_summary(rows: list[dict]) -> None:
    header = (
        f"{'Method':<22} | {'Acc%':>6} | {'Prec%':>6} | {'Rec%':>6} | "
        f"{'Spec%':>6} | {'F1%':>6} | {'FA-Area%':>8}"
    )
    print("\n================ ABLATION SUMMARY (bottle test set) ================")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['method']:<22} | {r['accuracy_pct']:>6.2f} | {r['precision_pct']:>6.2f} | "
            f"{r['recall_pct']:>6.2f} | {r['specificity_pct']:>6.2f} | {r['f1_pct']:>6.2f} | "
            f"{r['false_alarm_area_pct']:>8.2f}"
        )

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved ablation summary to: {SUMMARY_CSV}\n")


def run_ablation() -> None:
    golden = dict(load_images(GOLDEN_SET_DIR))
    mean_img, std_img = get_statistical_reference(
        BOTTLE_TRAIN_DIR, REFERENCE_CACHE, KERNEL_SIZE, BG_THRESHOLD
    )
    ref_aligned = preprocess_image(golden["pass_01.png"], KERNEL_SIZE, BG_THRESHOLD)

    save_visual_comparisons(golden, ref_aligned, mean_img, std_img)
    rows = evaluate_on_test_set(ref_aligned, mean_img, std_img)
    print_and_save_summary(rows)


if __name__ == "__main__":
    run_ablation()
