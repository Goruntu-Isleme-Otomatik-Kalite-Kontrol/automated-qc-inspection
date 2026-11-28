import argparse
import csv
from pathlib import Path
import cv2
import numpy as np
import yaml
from src.pipeline import run_pipeline


def _compute_pixel_metrics(
    pred_mask: np.ndarray, gt_mask_path: Path
) -> tuple[float, float] | None:
    """Compute pixel-level IoU and Dice against an MVTec ground_truth mask."""
    if not gt_mask_path.exists():
        return None

    gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        return None

    if pred_mask.shape != gt_mask.shape:
        pred_mask = cv2.resize(
            pred_mask,
            (gt_mask.shape[1], gt_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    pred_bool = pred_mask > 0
    gt_bool = gt_mask > 0

    intersection = float(np.logical_and(pred_bool, gt_bool).sum())
    union = float(np.logical_or(pred_bool, gt_bool).sum())
    denom = float(pred_bool.sum() + gt_bool.sum())

    iou = intersection / union if union > 0 else 1.0
    dice = (2.0 * intersection) / denom if denom > 0 else 1.0
    return iou, dice


def evaluate_full_dataset(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    test_root = Path(cfg["paths"]["test_root_dir"])
    if not test_root.exists():
        raise FileNotFoundError(f"Test dataset directory not found: {test_root}")

    gt_root = Path(
        cfg["paths"].get("ground_truth_dir", test_root.parent / "ground_truth")
    )
    cache_path = Path(cfg["paths"]["reference_cache"])

    tp = fp = tn = fn = 0
    category_stats = {}
    iou_scores = []
    dice_scores = []
    latencies_ms = []

    for category_dir in sorted(test_root.iterdir()):
        if not category_dir.is_dir():
            continue

        category = category_dir.name
        is_ground_truth_defect = category != "good"
        print(f"[{Path(config_path).stem}] Kategori işleniyor: {category} ...", flush=True)
        details = run_pipeline(
            config_path=config_path,
            input_dir_override=str(category_dir),
            save_outputs=False,
            return_details=True,
        )

        correct = 0
        for filename, info in details.items():
            status = info["status"]
            latencies_ms.append(info["latency_ms"])
            predicted_defect = status == "FAIL"

            if is_ground_truth_defect and predicted_defect:
                tp += 1
                correct += 1
            elif not is_ground_truth_defect and not predicted_defect:
                tn += 1
                correct += 1
            elif not is_ground_truth_defect and predicted_defect:
                fp += 1
            else:
                fn += 1

            # Compute pixel-level localization metrics on defective test samples
            if is_ground_truth_defect:
                gt_mask_path = gt_root / category / f"{Path(filename).stem}_mask.png"
                pixel_res = _compute_pixel_metrics(info["binary_mask"], gt_mask_path)
                if pixel_res is not None:
                    iou_scores.append(pixel_res[0])
                    dice_scores.append(pixel_res[1])

        total_cat = len(details)
        category_stats[category] = {
            "correct": correct,
            "total": total_cat,
            "accuracy": (correct / total_cat * 100) if total_cat else 0.0,
        }

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    avg_latency_ms = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0
    cache_size_mb = (
        cache_path.stat().st_size / (1024.0 * 1024.0) if cache_path.exists() else 0.0
    )

    print(f"\n=== Evaluation Results ({config_path}) ===")
    print("=== Per-Category Breakdown ===")
    for cat, stats in category_stats.items():
        print(f"  {cat:<15}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1f}%)")

    print("\n=== Confusion Matrix ===")
    print(f"  True Positives  (Defective -> FAIL): {tp}")
    print(f"  True Negatives  (Good      -> PASS): {tn}")
    print(f"  False Positives (Good      -> FAIL): {fp}")
    print(f"  False Negatives (Defective -> PASS): {fn}")

    print("\n=== Image & Pixel-Level Metrics ===")
    print(f"  Accuracy : {accuracy * 100:.2f}%")
    print(f"  Precision: {precision * 100:.2f}%")
    print(f"  Recall   : {recall * 100:.2f}%")
    print(f"  F1-Score : {f1 * 100:.2f}%")
    print(f"  Mean IoU : {mean_iou * 100:.2f}%")
    print(f"  Mean Dice: {mean_dice * 100:.2f}%")

    print("\n=== System Performance Profile ===")
    print(f"  Avg Latency : {avg_latency_ms:.2f} ms/img")
    print(f"  Throughput  : {fps:.1f} FPS")
    print(f"  Cache Size  : {cache_size_mb:.2f} MB\n")

    return {
        "config": config_path,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "avg_latency_ms": avg_latency_ms,
        "fps": fps,
        "cache_size_mb": cache_size_mb,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


DEFAULT_BENCHMARK_CONFIGS = [
    "configs/bottle_cv.yaml",
    "configs/bottle_nn.yaml",
    "configs/metal_nut_cv.yaml",
    "configs/metal_nut_nn.yaml",
    "configs/hazelnut_cv.yaml",
    "configs/hazelnut_nn.yaml",
]


def run_benchmark_suite(
    config_paths: list[str] | None = None,
    summary_csv: str = "results/benchmark_summary.csv",
) -> list[dict]:
    configs = config_paths or DEFAULT_BENCHMARK_CONFIGS
    summary_rows = []

    for i, cfg_path in enumerate(configs, start=1):
        print(f"\n[{i}/{len(configs)}] Benchmark çalıştırılıyor: {cfg_path}", flush=True)
        if not Path(cfg_path).exists():
            print(f"[SKIP] Config not found: {cfg_path}")
            continue
        metrics = evaluate_full_dataset(config_path=cfg_path)
        summary_rows.append(
            {
                "config": Path(cfg_path).stem,
                "accuracy_pct": round(metrics["accuracy"] * 100, 2),
                "precision_pct": round(metrics["precision"] * 100, 2),
                "recall_pct": round(metrics["recall"] * 100, 2),
                "f1_pct": round(metrics["f1"] * 100, 2),
                "mean_iou_pct": round(metrics["mean_iou"] * 100, 2),
                "mean_dice_pct": round(metrics["mean_dice"] * 100, 2),
                "latency_ms": round(metrics["avg_latency_ms"], 2),
                "fps": round(metrics["fps"], 1),
                "cache_mb": round(metrics["cache_size_mb"], 2),
                "tp": metrics["tp"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
            }
        )

    if summary_rows:
        out_csv = Path(summary_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

        print("\n================ 6-WAY BENCHMARK SUMMARY ================")
        header = (
            f"{'Config':<16} | {'Acc%':>6} | {'F1%':>6} | {'IoU%':>6} | "
            f"{'Dice%':>6} | {'ms/img':>7} | {'FPS':>6} | {'Cache(MB)':>9}"
        )
        print(header)
        print("-" * len(header))
        for r in summary_rows:
            print(
                f"{r['config']:<16} | {r['accuracy_pct']:>6.2f} | {r['f1_pct']:>6.2f} | "
                f"{r['mean_iou_pct']:>6.2f} | {r['mean_dice_pct']:>6.2f} | "
                f"{r['latency_ms']:>7.2f} | {r['fps']:>6.1f} | {r['cache_mb']:>9.2f}"
            )
        print(f"\nSaved full summary table to: {out_csv}\n")

    return summary_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate QC Inspection Pipeline")
    parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to single YAML config"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 6 benchmark configs and save results/benchmark_summary.csv",
    )
    args = parser.parse_args()

    if args.all:
        run_benchmark_suite()
    else:
        evaluate_full_dataset(config_path=args.config)