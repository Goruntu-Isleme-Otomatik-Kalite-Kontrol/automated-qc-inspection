from src.pipeline import run_pipeline

if __name__ == "__main__":
    summary = run_pipeline()
    for name, status in summary.items():
        print(f"{name}: {status}")