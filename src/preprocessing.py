import cv2
import numpy as np


def preprocess_image(
    image: np.ndarray,
    kernel_size: int = 9,
    bg_threshold: int = 210,
    color_space: str = "gray",
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # Segment bottle silhouette from bright background to compute centroid
    _, silhouette = cv2.threshold(
        blurred_gray, bg_threshold, 255, cv2.THRESH_BINARY_INV
    )
    moments = cv2.moments(silhouette)
    h, w = gray.shape

    if moments["m00"] > 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        dx = (w / 2.0) - cx
        dy = (h / 2.0) - cy
    else:
        dx, dy = 0.0, 0.0

    shift_matrix = np.float32([[1, 0, dx], [0, 1, dy]])

    if color_space.lower() == "lab":
        blurred_bgr = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        lab = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2LAB)
        return cv2.warpAffine(
            lab, shift_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE
        )

    return cv2.warpAffine(
        blurred_gray, shift_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE
    )