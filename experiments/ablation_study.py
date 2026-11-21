from pathlib import Path
import cv2
import numpy as np
from src.acquisition import load_images
from src.preprocessing import preprocess_image
from src.pipeline import get_statistical_reference


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


def run_ablation():
    out_dir = Path("docs/images")
    out_dir.mkdir(parents=True, exist_ok=True)

    golden = dict(load_images("data/golden_set"))
    mean_img, std_img = get_statistical_reference(
        "data/raw/train/good", "outputs/reference_model.npz", 9, 210
    )

    ref_aligned = preprocess_image(golden["pass_01.png"], 9, 210)

    for sample_name in ["pass_02.png", "fail_01.png"]:
        img = golden[sample_name]
        aligned = preprocess_image(img, 9, 210)

        m1 = method_1_global_threshold(aligned)
        m2 = method_2_canny_roi(aligned)
        m3 = method_3_single_reference(aligned, ref_aligned)
        m4 = method_4_statistical_model(aligned, mean_img, std_img)

        panels = []
        titles = [
            "1. Global Thresh (Fail)",
            "2. Canny + ROI (Fail)",
            "3. Single Ref (Fail)",
            "4. Mean+Std Model (Pass)",
        ]
        for title, mask in zip(titles, [m1, m2, m3, m4]):
            vis = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
            vis[mask > 0] = (0, 0, 255)  # Highlight flagged defect pixels in red
            vis = cv2.resize(vis, (320, 320))
            cv2.putText(vis, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
            panels.append(vis)

        grid = np.hstack(panels)
        save_path = out_dir / f"ablation_{sample_name}"
        cv2.imwrite(str(save_path), grid)
        print(f"Saved visual comparison: {save_path}")


if __name__ == "__main__":
    run_ablation()