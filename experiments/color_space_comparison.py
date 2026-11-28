"""Grayscale vs. CIE LAB: does the color space of the classical track matter?

For every category the classical configuration (configs/<category>_cv.yaml) is evaluated twice.
Only `preprocessing.color_space` changes between the two runs; every other parameter is kept
identical, so any difference in the results is caused by the color space alone.
"""
import copy
import csv
from pathlib import Path
import matplotlib

matplotlib.use("Agg")  # render to file without opening a window
import matplotlib.pyplot as plt
import yaml
from src.evaluate import evaluate_full_dataset

CATEGORIES = ["bottle", "metal_nut", "hazelnut"]
COLOR_SPACES = ["gray", "lab"]
CONFIG_TEMPLATE = "configs/{category}_cv.yaml"
VARIANT_CONFIG_DIR = Path("data/cache/color_space_configs")  # generated, not versioned
SUMMARY_CSV = Path("results/color_space_summary.csv")
PER_DEFECT_CSV = Path("results/color_space_per_defect.csv")
FIGURE_PATH = Path("docs/images/color_space_comparison.png")


def make_variant_config(category: str, color_space: str) -> Path:
    """Copy the category's classical config, changing only the color space and the cache file."""
    with open(CONFIG_TEMPLATE.format(category=category), "r", encoding="utf-8") as f:
        cfg = copy.deepcopy(yaml.safe_load(f))

    cfg["preprocessing"]["color_space"] = color_space
    # Separate cache per color space, otherwise a gray run would reuse a LAB reference model
    cfg["paths"]["reference_cache"] = f"data/cache/{category}_cv_{color_space}.npz"

    VARIANT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    variant_path = VARIANT_CONFIG_DIR / f"{category}_cv_{color_space}.yaml"
    with open(variant_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return variant_path


def run_comparison() -> tuple[list[dict], list[dict]]:
    summary_rows, per_defect_rows = [], []

    for category in CATEGORIES:
        for color_space in COLOR_SPACES:
            print(f"\n##### {category} / {color_space} #####", flush=True)
            metrics = evaluate_full_dataset(str(make_variant_config(category, color_space)))

            summary_rows.append(
                {
                    "category": category,
                    "color_space": color_space,
                    "accuracy_pct": round(metrics["accuracy"] * 100, 2),
                    "precision_pct": round(metrics["precision"] * 100, 2),
                    "recall_pct": round(metrics["recall"] * 100, 2),
                    "f1_pct": round(metrics["f1"] * 100, 2),
                    "mean_iou_pct": round(metrics["mean_iou"] * 100, 2),
                    "latency_ms": round(metrics["avg_latency_ms"], 2),
                }
            )
            for defect_type, stats in metrics["category_stats"].items():
                per_defect_rows.append(
                    {
                        "category": category,
                        "color_space": color_space,
                        "defect_type": defect_type,
                        "correct": stats["correct"],
                        "total": stats["total"],
                        "accuracy_pct": round(stats["accuracy"], 2),
                    }
                )

    return summary_rows, per_defect_rows


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def plot_per_defect(summary_rows: list[dict], per_defect_rows: list[dict]) -> None:
    """One panel per product: detection accuracy per defect type, gray vs. LAB side by side."""
    fig, axes = plt.subplots(1, len(CATEGORIES), figsize=(16, 4.8))
    bar_width = 0.38
    colors = {"gray": "#8c8c8c", "lab": "#2f7ed8"}

    for ax, category in zip(axes, CATEGORIES):
        rows = [r for r in per_defect_rows if r["category"] == category]
        defect_types = sorted({r["defect_type"] for r in rows})
        f1 = {r["color_space"]: r["f1_pct"] for r in summary_rows if r["category"] == category}

        for offset, color_space in zip((-bar_width / 2, bar_width / 2), COLOR_SPACES):
            values = [
                next(r["accuracy_pct"] for r in rows
                     if r["defect_type"] == d and r["color_space"] == color_space)
                for d in defect_types
            ]
            positions = [i + offset for i in range(len(defect_types))]
            ax.bar(positions, values, bar_width, label=f"{color_space.upper()} (F1 {f1[color_space]:.1f}%)",
                   color=colors[color_space])

        ax.set_title(category)
        ax.set_xticks(range(len(defect_types)))
        ax.set_xticklabels(defect_types, rotation=30, ha="right")
        ax.set_ylim(0, 125)
        ax.set_yticks(range(0, 101, 20))
        ax.set_ylabel("Correctly classified (%)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper center", ncol=2, fontsize=8)

    fig.suptitle("Classical track: grayscale vs. CIE LAB per defect type")
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150)
    print(f"Saved: {FIGURE_PATH}")


def print_summary(summary_rows: list[dict]) -> None:
    header = f"{'Category':<10} | {'Space':<5} | {'Acc%':>6} | {'F1%':>6} | {'IoU%':>6} | {'ms/img':>7}"
    print("\n========== COLOR SPACE COMPARISON (classical track) ==========")
    print(header)
    print("-" * len(header))
    for r in summary_rows:
        print(
            f"{r['category']:<10} | {r['color_space']:<5} | {r['accuracy_pct']:>6.2f} | "
            f"{r['f1_pct']:>6.2f} | {r['mean_iou_pct']:>6.2f} | {r['latency_ms']:>7.2f}"
        )


if __name__ == "__main__":
    summary, per_defect = run_comparison()
    print_summary(summary)
    save_csv(summary, SUMMARY_CSV)
    save_csv(per_defect, PER_DEFECT_CSV)
    plot_per_defect(summary, per_defect)
