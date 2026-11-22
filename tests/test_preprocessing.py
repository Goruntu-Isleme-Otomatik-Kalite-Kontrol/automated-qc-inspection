import cv2
import numpy as np
from src.preprocessing import preprocess_image


def _create_offcenter_bottle_image(
    size: int = 300, center: tuple[int, int] = (110, 190), radius: int = 40
) -> np.ndarray:
    """Generate a synthetic dark circular object off-center on a white background."""
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    cv2.circle(img, center, radius, (30, 45, 80), -1)
    return img


def test_preprocess_grayscale_shape_and_dtype():
    img = _create_offcenter_bottle_image()
    result = preprocess_image(img, kernel_size=9, bg_threshold=210, color_space="gray")

    assert result.ndim == 2
    assert result.shape == (300, 300)
    assert result.dtype == np.uint8


def test_preprocess_lab_multichannel_shape():
    img = _create_offcenter_bottle_image()
    result = preprocess_image(img, kernel_size=9, bg_threshold=210, color_space="lab")

    assert result.ndim == 3
    assert result.shape == (300, 300, 3)
    assert result.dtype == np.uint8


def test_centroid_alignment_centers_object():
    img = _create_offcenter_bottle_image(size=300, center=(100, 200), radius=45)
    aligned = preprocess_image(img, kernel_size=5, bg_threshold=210, color_space="gray")

    _, silhouette = cv2.threshold(aligned, 210, 255, cv2.THRESH_BINARY_INV)
    moments = cv2.moments(silhouette)
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    # After alignment, the object's centroid must sit at the frame center (150, 150)
    assert abs(cx - 150.0) < 2.0
    assert abs(cy - 150.0) < 2.0