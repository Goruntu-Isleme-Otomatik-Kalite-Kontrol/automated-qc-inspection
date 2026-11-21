import argparse
from src.pipeline import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated QC Inspection")
    parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to YAML config"
    )
    args = parser.parse_args()

    results = run_pipeline(args.config)
    for filename, status in sorted(results.items()):
        print(f"{filename}: {status}")