import cv2
import numpy as np

def preprocess_image(
    image: np.ndarray,
    kernel_size: int,
    bg_threshold: int,
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # Segment bottle to find its center of mass
    _, mask = cv2.threshold(blurred, bg_threshold, 255, cv2.THRESH_BINARY_INV)
    m = cv2.moments(mask)
    if m["m00"] > 0:
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        h, w = blurred.shape
        shift_x = (w // 2) - cx
        shift_y = (h // 2) - cy
        matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        blurred = cv2.warpAffine(blurred, matrix, (w, h), borderValue=255)

    return blurred