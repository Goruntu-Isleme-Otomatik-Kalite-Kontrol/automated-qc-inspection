import csv
import time
from pathlib import Path
import cv2
import numpy as np
import yaml
from src.acquisition import load_images
from src.preprocessing import preprocess_image
from src.inspection import detect_defects
from src.postprocessing import annotate_and_save
from src.nn_detector import (
    ResNetPatchFeatureExtractor,
    build_memory_bank,
    save_memory_bank,
    load_memory_bank,
    detect_nn_defects,
)


def get_statistical_reference(
    ref_dir: str,
    cache_path: str,
    kernel_size: int,
    bg_threshold: int,
    color_space: str = "gray",
    bg_mode: str = "auto",
    use_clahe: bool = False,
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    cache_file = Path(cache_path)
    if use_cache and cache_file.exists():
        data = np.load(cache_file)
        return data["mean_img"], data["std_img"]

    stack = []
    for _, img in load_images(ref_dir):
        processed = preprocess_image(
            img,
            kernel_size=kernel_size,
            bg_threshold=bg_threshold,
            color_space=color_space,
            bg_mode=bg_mode,
            use_clahe=use_clahe,
        )
        stack.append(processed.astype(np.float32))

    if not stack:
        raise FileNotFoundError(f"No reference images found in: {ref_dir}")

    arr = np.stack(stack, axis=0)
    mean_img = np.mean(arr, axis=0)
    std_img = np.std(arr, axis=0)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, mean_img=mean_img, std_img=std_img)
    return mean_img, std_img


def get_nn_reference(
    ref_dir: str,
    cache_path: str,
    extractor: ResNetPatchFeatureExtractor,
    kernel_size: int = 9,
    bg_threshold: int = 210,
    bg_mode: str = "auto",
    use_clahe: bool = True,
    target_size: tuple[int, int] = (224, 224),
    max_patches: int = 15000,
    use_cache: bool = True,
) -> np.ndarray:
    cache_file = Path(cache_path)
    if use_cache and cache_file.exists():
        return load_memory_bank(cache_file)

    train_rgb = []
    for _, img in load_images(ref_dir):
        processed = preprocess_image(
            img,
            kernel_size=kernel_size,
            bg_threshold=bg_threshold,
            color_space="rgb",
            bg_mode=bg_mode,
            use_clahe=use_clahe,
            target_size=target_size,
        )
        train_rgb.append(processed)

    if not train_rgb:
        raise FileNotFoundError(f"No reference images found in: {ref_dir}")

    memory_bank = build_memory_bank(
        train_rgb, extractor=extractor, max_patches=max_patches
    )
    save_memory_bank(cache_file, memory_bank)
    return memory_bank


def _scale_boxes(
    boxes: list[tuple[int, int, int, int]],
    src_size: tuple[int, int],
    dst_size: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    """Rescale (x, y, w, h) boxes from preprocessed size back to original image size."""
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    scale_x = dst_w / float(src_w)
    scale_y = dst_h / float(src_h)
    return [
        (
            int(round(x * scale_x)),
            int(round(y * scale_y)),
            int(round(w * scale_x)),
            int(round(h * scale_y)),
        )
        for (x, y, w, h) in boxes
    ]


def run_pipeline(
    config_path: str = "configs/default.yaml",
    input_dir_override: str | None = None,
    save_outputs: bool = True,
    return_details: bool = False,
) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    model_type = model_cfg.get("type", "statistical_cv").lower()
    prep_cfg = cfg["preprocessing"]
    insp_cfg = cfg["inspection"]

    bg_mode = prep_cfg.get("bg_mode", "auto")
    use_clahe = prep_cfg.get("use_clahe", False)

    if model_type == "nn_resnet":
        target_wh = tuple(prep_cfg.get("target_size", [224, 224]))
        extractor = ResNetPatchFeatureExtractor(device=model_cfg.get("device"))
        memory_bank = get_nn_reference(
            cfg["paths"]["reference_dir"],
            cfg["paths"]["reference_cache"],
            extractor=extractor,
            kernel_size=prep_cfg["blur_kernel_size"],
            bg_threshold=prep_cfg.get("bg_threshold", 210),
            bg_mode=bg_mode,
            use_clahe=use_clahe,
            target_size=target_wh,
            max_patches=model_cfg.get("max_patches", 15000),
        )
    else:
        color_space = prep_cfg.get("color_space", "gray")
        mean_img, std_img = get_statistical_reference(
            cfg["paths"]["reference_dir"],
            cfg["paths"]["reference_cache"],
            prep_cfg["blur_kernel_size"],
            prep_cfg["bg_threshold"],
            color_space=color_space,
            bg_mode=bg_mode,
            use_clahe=use_clahe,
        )

    target_dir = input_dir_override or cfg["paths"]["input_dir"]
    out_dir = Path(cfg["paths"]["output_dir"])
    results = {}
    detailed_results = {}
    report_rows = []

    for filename, image in load_images(target_dir):
        orig_h, orig_w = image.shape[:2]
        t_start = time.perf_counter()

        if model_type == "nn_resnet":
            processed = preprocess_image(
                image,
                kernel_size=prep_cfg["blur_kernel_size"],
                bg_threshold=prep_cfg.get("bg_threshold", 210),
                color_space="rgb",
                bg_mode=bg_mode,
                use_clahe=use_clahe,
                target_size=target_wh,
            )
            nn_out = detect_nn_defects(
                processed,
                memory_bank=memory_bank,
                extractor=extractor,
                anomaly_threshold=insp_cfg.get("anomaly_threshold", 0.45),
                min_defect_area_px=insp_cfg.get("min_defect_area_px", 120),
                morph_kernel_size=insp_cfg.get("morph_kernel_size", 5),
            )
            is_defective = nn_out["label"] == "FAIL"
            boxes = _scale_boxes(nn_out["boxes"], target_wh, (orig_w, orig_h))
            full_mask = cv2.resize(
                nn_out["binary_mask"], (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
            )
            heatmap = cv2.resize(
                nn_out["heatmap"], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
            )
        else:
            processed = preprocess_image(
                image,
                kernel_size=prep_cfg["blur_kernel_size"],
                bg_threshold=prep_cfg["bg_threshold"],
                color_space=color_space,
                bg_mode=bg_mode,
                use_clahe=use_clahe,
            )
            is_defective, boxes = detect_defects(
                processed,
                mean_img,
                std_img,
                insp_cfg["std_multiplier"],
                insp_cfg["min_abs_diff"],
                insp_cfg["min_defect_area_px"],
            )
            # Reconstruct binary mask from boxes for unified pixel-level IoU evaluation
            full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            for x, y, w, h in boxes:
                full_mask[y : y + h, x : x + w] = 255
            heatmap = None

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        status = "FAIL" if is_defective else "PASS"
        results[filename] = status

        if return_details:
            detailed_results[filename] = {
                "status": status,
                "boxes": boxes,
                "binary_mask": full_mask,
                "heatmap": heatmap,
                "latency_ms": latency_ms,
            }

        total_box_area = sum(w * h for (_, _, w, h) in boxes)
        report_rows.append(
            {
                "filename": filename,
                "status": status,
                "defect_count": len(boxes),
                "total_bbox_area_px": total_box_area,
                "bounding_boxes": str(boxes),
                "latency_ms": round(latency_ms, 2),
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
                    "latency_ms",
                ],
            )
            writer.writeheader()
            writer.writerows(report_rows)

    return detailed_results if return_details else results