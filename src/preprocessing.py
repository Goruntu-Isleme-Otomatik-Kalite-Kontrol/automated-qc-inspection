import cv2
import numpy as np

def preprocess_image(image: np.ndarray, kernel_size: int) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)