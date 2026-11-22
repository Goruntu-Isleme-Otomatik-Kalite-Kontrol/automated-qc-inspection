import argparse
from pathlib import Path
import yaml
from src.pipeline import run_pipeline


def evaluate_full_dataset(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    test_root = Path(cfg["paths"]["test_root_dir"])
    if not test_root.exists():
        raise FileNotFoundError(f"Test dataset directory not found: {test_root}")

    tp = fp = tn = fn = 0
    category_stats = {}

    for category_dir in sorted(test_root.iterdir()):
        if not category_dir.is_dir():
            continue

        category = category_dir.name
        is_ground_truth_defect = category != "good"
        preds = run_pipeline(
            config_path=config_path,
            input_dir_override=str(category_dir),
            save_outputs=False,
        )

        correct = 0
        for status in preds.values():
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

        total_cat = len(preds)
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

    print(f"\n=== Evaluation Results ({config_path}) ===")
    print("=== Per-Category Breakdown ===")
    for cat, stats in category_stats.items():
        print(f"  {cat:<15}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1f}%)")

    print("\n=== Confusion Matrix ===")
    print(f"  True Positives  (Defective -> FAIL): {tp}")
    print(f"  True Negatives  (Good      -> PASS): {tn}")
    print(f"  False Positives (Good      -> FAIL): {fp}")
    print(f"  False Negatives (Defective -> PASS): {fn}")

    print("\n=== Overall Metrics ===")
    print(f"  Accuracy : {accuracy * 100:.2f}%")
    print(f"  Precision: {precision * 100:.2f}%")
    print(f"  Recall   : {recall * 100:.2f}%")
    print(f"  F1-Score : {f1 * 100:.2f}%\n")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate QC Inspection Pipeline")
    parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to YAML config"
    )
    args = parser.parse_args()
    evaluate_full_dataset(config_path=args.config)