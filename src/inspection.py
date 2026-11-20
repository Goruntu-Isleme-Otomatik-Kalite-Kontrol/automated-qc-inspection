import cv2
import numpy as np

def detect_defects(processed_img: np.ndarray, thresh_val: int, min_area: int):
    _, mask = cv2.threshold(processed_img, thresh_val, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) >= min_area]
    return len(boxes) > 0, boxes