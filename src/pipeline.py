from pathlib import Path
import numpy as np
import yaml
from src.acquisition import load_images
from src.preprocessing import preprocess_image
from src.inspection import detect_defects
from src.postprocessing import annotate_and_save

def build_statistical_reference(
    ref_dir: str, kernel_size: int, bg_threshold: int
) -> tuple[np.ndarray, np.ndarray]:
    stack = []
    for _, img in load_images(ref_dir):
        processed = preprocess_image(img, kernel_size, bg_threshold)
        stack.append(processed.astype(np.float32))
    if not stack:
        raise FileNotFoundError(f"No reference images found in: {ref_dir}")
    arr = np.stack(stack, axis=0)
    return np.mean(arr, axis=0), np.std(arr, axis=0)

def run_pipeline(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    prep_cfg = cfg["preprocessing"]
    insp_cfg = cfg["inspection"]

    mean_img, std_img = build_statistical_reference(
        cfg["paths"]["reference_dir"],
        prep_cfg["blur_kernel_size"],
        prep_cfg["bg_threshold"],
    )

    results = {}
    out_dir = Path(cfg["paths"]["output_dir"])

    for filename, image in load_images(cfg["paths"]["input_dir"]):
        processed = preprocess_image(
            image, prep_cfg["blur_kernel_size"], prep_cfg["bg_threshold"]
        )
        is_defective, boxes = detect_defects(
            processed,
            mean_img,
            std_img,
            insp_cfg["std_multiplier"],
            insp_cfg["min_abs_diff"],
            insp_cfg["min_defect_area_px"],
        )
        annotate_and_save(image, boxes, is_defective, out_dir / filename)
        results[filename] = "FAIL" if is_defective else "PASS"

    return results