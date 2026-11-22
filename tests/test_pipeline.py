from pathlib import Path
import yaml
from src.pipeline import run_pipeline


def test_golden_set_safety_and_artifacts():
    with open("configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    results = run_pipeline("configs/default.yaml")
    assert len(results) > 0, "Pipeline returned no results"

    pass_results = [status for name, status in results.items() if name.startswith("pass_")]
    fail_results = [status for name, status in results.items() if name.startswith("fail_")]

    assert len(fail_results) > 0 and len(pass_results) > 0

    # Manufacturing safety invariant: zero escaped defects on known bad samples
    for filename, status in results.items():
        if filename.startswith("fail_"):
            assert status == "FAIL", f"Escaped defect on {filename}"

    # Sanity bound: ensure the detector does not blindly classify all good bottles as FAIL
    pass_rate = pass_results.count("PASS") / len(pass_results)
    assert pass_rate >= 0.5, f"Excessive false positives on golden pass set: {pass_rate:.0%}"

    # Verify Sprint 3 artifacts
    assert Path(cfg["paths"]["reference_cache"]).exists(), "Reference cache .npz not created"
    assert Path(cfg["paths"]["report_csv"]).exists(), "Audit report.csv not created"