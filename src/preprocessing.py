import cv2
import numpy as np


def _segment_silhouette(
    blurred_gray: np.ndarray,
    bg_mode: str = "auto",
    bg_threshold: int = 210,
) -> np.ndarray:
    """Segment the foreground object from either bright or dark backgrounds."""
    if bg_mode.lower() == "fixed":
        _, silhouette = cv2.threshold(
            blurred_gray, bg_threshold, 255, cv2.THRESH_BINARY_INV
        )
        return silhouette

    # Sample a 15-pixel border frame to detect whether background is light or dark
    h, w = blurred_gray.shape
    border = max(5, min(15, h // 20, w // 20))
    border_pixels = np.concatenate(
        [
            blurred_gray[:border, :].ravel(),
            blurred_gray[-border:, :].ravel(),
            blurred_gray[:, :border].ravel(),
            blurred_gray[:, -border:].ravel(),
        ]
    )
    is_light_bg = float(np.mean(border_pixels)) > 127.0

    thresh_type = (
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        if is_light_bg
        else cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    _, silhouette = cv2.threshold(blurred_gray, 0, 255, thresh_type)
    return silhouette


def _apply_clahe_bgr(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Balance harsh specular glare while preserving natural color channels."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_chan)
    merged = cv2.merge((l_eq, a_chan, b_chan))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def preprocess_image(
    image: np.ndarray,
    kernel_size: int = 9,
    bg_threshold: int = 210,
    color_space: str = "gray",
    bg_mode: str = "auto",
    use_clahe: bool = False,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # 1. Segment object silhouette (auto-handles bright bottle & dark metal_nut/hazelnut)
    silhouette = _segment_silhouette(
        blurred_gray, bg_mode=bg_mode, bg_threshold=bg_threshold
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
    space = color_space.lower()

    # 2. Shift & convert to target color space
    if space == "rgb":
        # Keep fine surface textures sharp for ResNet-18 (only blur was used for centroid)
        working_bgr = _apply_clahe_bgr(image) if use_clahe else image
        aligned_bgr = cv2.warpAffine(
            working_bgr, shift_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE
        )
        result = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    elif space == "lab":
        working_bgr = _apply_clahe_bgr(image) if use_clahe else image
        blurred_bgr = cv2.GaussianBlur(working_bgr, (kernel_size, kernel_size), 0)
        lab = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2LAB)
        result = cv2.warpAffine(
            lab, shift_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE
        )
    else:
        result = cv2.warpAffine(
            blurred_gray, shift_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE
        )

    # 3. Optional standardization resize (e.g., (224, 224) for ResNet-18)
    if target_size is not None:
        result = cv2.resize(result, target_size, interpolation=cv2.INTER_AREA)

    return result