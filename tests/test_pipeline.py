from src.pipeline import run_pipeline

def test_golden_set_classification():
    results = run_pipeline("configs/default.yaml")
    assert len(results) > 0
    for filename, status in results.items():
        if filename.startswith("pass_"):
            assert status == "PASS", f"False positive on {filename}"
        elif filename.startswith("fail_"):
            assert status == "FAIL", f"Missed defect on {filename}"