import csv
from pathlib import Path
import numpy as np
import yaml
from src.acquisition import load_images
from src.preprocessing import preprocess_image
from src.inspection import detect_defects
from src.postprocessing import annotate_and_save


def get_statistical_reference(
    ref_dir: str,
    cache_path: str,
    kernel_size: int,
    bg_threshold: int,
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    cache_file = Path(cache_path)
    if use_cache and cache_file.exists():
        data = np.load(cache_file)
        return data["mean_img"], data["std_img"]

    stack = []
    for _, img in load_images(ref_dir):
        processed = preprocess_image(img, kernel_size, bg_threshold)
        stack.append(processed.astype(np.float32))

    if not stack:
        raise FileNotFoundError(f"No reference images found in: {ref_dir}")

    arr = np.stack(stack, axis=0)
    mean_img = np.mean(arr, axis=0)
    std_img = np.std(arr, axis=0)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, mean_img=mean_img, std_img=std_img)
    return mean_img, std_img


def run_pipeline(
    config_path: str = "configs/default.yaml",
    input_dir_override: str | None = None,
    save_outputs: bool = True,
) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    prep_cfg = cfg["preprocessing"]
    insp_cfg = cfg["inspection"]

    mean_img, std_img = get_statistical_reference(
        cfg["paths"]["reference_dir"],
        cfg["paths"]["reference_cache"],
        prep_cfg["blur_kernel_size"],
        prep_cfg["bg_threshold"],
    )

    target_dir = input_dir_override or cfg["paths"]["input_dir"]
    out_dir = Path(cfg["paths"]["output_dir"])
    results = {}
    report_rows = []

    for filename, image in load_images(target_dir):
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
        status = "FAIL" if is_defective else "PASS"
        results[filename] = status

        total_box_area = sum(w * h for (_, _, w, h) in boxes)
        report_rows.append(
            {
                "filename": filename,
                "status": status,
                "defect_count": len(boxes),
                "total_bbox_area_px": total_box_area,
                "bounding_boxes": str(boxes),
            }
        )

        if save_outputs:
            annotate_and_save(image, boxes, is_defective, out_dir / filename)

    if save_outputs:
        report_path = Path(cfg["paths"]["report_csv"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "filename",
                    "status",
                    "defect_count",
                    "total_bbox_area_px",
                    "bounding_boxes",
                ],
            )
            writer.writeheader()
            writer.writerows(report_rows)

    return results