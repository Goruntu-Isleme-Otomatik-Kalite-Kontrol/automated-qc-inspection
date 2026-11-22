import cv2
import numpy as np


def detect_defects(
    aligned_img: np.ndarray,
    mean_img: np.ndarray,
    std_img: np.ndarray,
    std_multiplier: float = 2.8,
    min_abs_diff: float = 25.0,
    min_defect_area_px: int = 150,
) -> tuple[bool, list[tuple[int, int, int, int]]]:
    diff = np.abs(aligned_img.astype(np.float32) - mean_img)
    tolerance = (std_multiplier * std_img) + min_abs_diff
    exceeded = diff > tolerance

    # Collapse 3-channel (H, W, 3) color deviations into a single 2D binary mask
    if exceeded.ndim == 3:
        exceeded = np.any(exceeded, axis=2)

    binary_mask = exceeded.astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    defect_boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_defect_area_px:
            x, y, w, h = cv2.boundingRect(cnt)
            defect_boxes.append((x, y, w, h))

    return len(defect_boxes) > 0, defect_boxes