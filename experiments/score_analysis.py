"""Image-level score analysis for all six benchmark configurations.

1. AUROC: threshold-independent measure of how well a configuration separates defective
   from defect-free images (the standard image-level metric on MVTec AD).
2. Threshold sensitivity: how the F1-score of the hybrid track changes with the anomaly threshold.
3. Confusion matrices: visual form of the TP / TN / FP / FN counts in results/benchmark_summary.csv.
"""
import csv
from pathlib import Path
import matplotlib

matplotlib.use("Agg")  # render to file without opening a window
import matplotlib.pyplot as plt
import numpy as np
import yaml
from src.pipeline import run_pipeline

CATEGORIES = ["bottle", "metal_nut", "hazelnut"]
TRACKS = ["cv", "nn"]
BENCHMARK_CSV = Path("results/benchmark_summary.csv")
AUROC_CSV = Path("results/auroc_summary.csv")
SWEEP_FIGURE = Path("docs/images/threshold_sweep.png")
CONFUSION_FIGURE = Path("docs/images/confusion_matrices.png")


def collect_scores(config_path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run a configuration on its test set and return one anomaly score + true label per image.

    Hybrid track: peak of the anomaly heatmap (max patch distance).
    Classical track: number of pixels inside the detected defect regions.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    test_root = Path(cfg["paths"]["test_root_dir"])
    scores, labels = [], []
    for category_dir in sorted(test_root.iterdir()):
        if not category_dir.is_dir():
            continue
        print(f"[{Path(config_path).stem}] Kategori işleniyor: {category_dir.name} ...", flush=True)
        details = run_pipeline(
            config_path=config_path,
            input_dir_override=str(category_dir),
            save_outputs=False,
            return_details=True,
        )
        for info in details.values():
            if info["heatmap"] is not None:
                score = float(np.max(info["heatmap"]))
            else:
                score = float(np.count_nonzero(info["binary_mask"]))
            scores.append(score)
            labels.append(int(category_dir.name != "good"))

    return np.array(scores), np.array(labels), cfg


def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Probability that a random defective image scores higher than a random defect-free one."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    greater = np.sum(pos[:, None] > neg[None, :])
    ties = np.sum(pos[:, None] == neg[None, :])
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def f1_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    predicted = scores > threshold
    tp = np.sum(predicted & (labels == 1))
    fp = np.sum(predicted & (labels == 0))
    fn = np.sum(~predicted & (labels == 1))
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 0.0


def plot_threshold_sweep(sweep_data: dict) -> None:
    fig, axes = plt.subplots(1, len(CATEGORIES), figsize=(16, 4.5))
    for ax, category in zip(axes, CATEGORIES):
        scores, labels, config_threshold = sweep_data[category]
        thresholds = np.linspace(scores.min(), scores.max(), 200)
        f1_values = [100 * f1_at_threshold(scores, labels, t) for t in thresholds]
        best = int(np.argmax(f1_values))

        ax.plot(thresholds, f1_values, color="#2f7ed8", linewidth=2)
        ax.axvline(config_threshold, color="#d9534f", linestyle="--",
                   label=f"Config threshold ({config_threshold:.2f})")
        ax.scatter([thresholds[best]], [f1_values[best]], color="#2e7d32", zorder=3,
                   label=f"Best: {thresholds[best]:.2f} (F1 {f1_values[best]:.1f}%)")
        ax.set_title(f"{category}_nn")
        ax.set_xlabel("Anomaly threshold (peak heatmap score)")
        ax.set_ylabel("F1-score (%)")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)

    fig.suptitle("Hybrid track: F1-score vs. anomaly threshold (score-only decision, no area filter)")
    fig.tight_layout()
    SWEEP_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SWEEP_FIGURE, dpi=150)
    print(f"Saved: {SWEEP_FIGURE}")


def plot_confusion_matrices() -> None:
    with open(BENCHMARK_CSV, "r", encoding="utf-8") as f:
        rows = {r["config"]: r for r in csv.DictReader(f)}

    fig, axes = plt.subplots(len(CATEGORIES), len(TRACKS), figsize=(8, 11))
    for i, category in enumerate(CATEGORIES):
        for j, track in enumerate(TRACKS):
            r = rows[f"{category}_{track}"]
            matrix = np.array([[int(r["tn"]), int(r["fp"])], [int(r["fn"]), int(r["tp"])]])
            ax = axes[i, j]
            ax.imshow(matrix, cmap="Blues")
            for (y, x), value in np.ndenumerate(matrix):
                color = "white" if value > matrix.max() / 2 else "black"
                ax.text(x, y, str(value), ha="center", va="center", fontsize=14, color=color)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["PASS", "FAIL"])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["good", "defective"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"{category}_{track}  (F1 {float(r['f1_pct']):.2f}%)")

    fig.suptitle("Confusion matrices (from results/benchmark_summary.csv)")
    fig.tight_layout()
    CONFUSION_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CONFUSION_FIGURE, dpi=150)
    print(f"Saved: {CONFUSION_FIGURE}")


def main() -> None:
    auroc_rows = []
    sweep_data = {}

    for category in CATEGORIES:
        for track in TRACKS:
            config_path = f"configs/{category}_{track}.yaml"
            scores, labels, cfg = collect_scores(config_path)
            row = {
                "config": f"{category}_{track}",
                "score": "peak anomaly score" if track == "nn" else "defect area (px)",
                "auroc_pct": round(100 * compute_auroc(scores, labels), 2),
            }
            if track == "nn":
                config_threshold = float(cfg["inspection"]["anomaly_threshold"])
                sweep_data[category] = (scores, labels, config_threshold)
            auroc_rows.append(row)

    print("\n========== IMAGE-LEVEL AUROC ==========")
    for r in auroc_rows:
        print(f"  {r['config']:<14}: {r['auroc_pct']:6.2f}%   ({r['score']})")

    AUROC_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(AUROC_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(auroc_rows[0].keys()))
        writer.writeheader()
        writer.writerows(auroc_rows)
    print(f"\nSaved: {AUROC_CSV}")

    plot_threshold_sweep(sweep_data)
    plot_confusion_matrices()


if __name__ == "__main__":
    main()
