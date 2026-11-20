from pathlib import Path
from src.pipeline import run_pipeline

def test_pipeline_runs_and_creates_outputs():
    results = run_pipeline("configs/default.yaml")
    assert len(results) > 0
    for filename in results:
        assert (Path("outputs") / filename).exists()