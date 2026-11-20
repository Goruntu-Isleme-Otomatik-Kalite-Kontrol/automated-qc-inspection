from pathlib import Path
import yaml
from src.acquisition import load_images
from src.preprocessing import preprocess_image
from src.inspection import detect_defects
from src.postprocessing import annotate_and_save

def run_pipeline(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    results = {}
    out_dir = Path(cfg["paths"]["output_dir"])

    for filename, image in load_images(cfg["paths"]["input_dir"]):
        processed = preprocess_image(image, cfg["preprocessing"]["blur_kernel_size"])
        is_defective, boxes = detect_defects(
            processed,
            cfg["inspection"]["threshold_value"],
            cfg["inspection"]["min_defect_area_px"],
        )
        annotate_and_save(image, boxes, is_defective, out_dir / filename)
        results[filename] = "FAIL" if is_defective else "PASS"

    return results