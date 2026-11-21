import cv2
import numpy as np

def detect_defects(
    test_img: np.ndarray,
    mean_img: np.ndarray,
    std_img: np.ndarray,
    std_multiplier: float,
    min_abs_diff: int,
    min_area: int,
) -> tuple[bool, list]:
    diff = np.abs(test_img.astype(np.float32) - mean_img)
    tolerance = (std_multiplier * std_img) + min_abs_diff
    anomaly_mask = (diff > tolerance).astype(np.uint8) * 255

    # Remove isolated pixel noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = [
        cv2.boundingRect(cnt)
        for cnt in contours
        if cv2.contourArea(cnt) >= min_area
    ]
    return len(boxes) > 0, boxes