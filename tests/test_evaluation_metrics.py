"""
Unit tests for FruitVision AI held-out test evaluation reports,
benchmark latency outputs, and artifact generation.
"""

import json
from pathlib import Path

REPORTS_DIR = Path("reports")
CLASSIFICATION_REPORT = REPORTS_DIR / "classification_report.json"
BENCHMARK_REPORT = REPORTS_DIR / "benchmark_report.json"
CONFUSION_MATRIX_IMG = REPORTS_DIR / "confusion_matrix.png"
TRAINING_CURVES_IMG = REPORTS_DIR / "training_curves.png"

def test_evaluation_artifacts_exist():
    """Verify that all empirical evaluation reports and visualizations were generated."""
    assert CLASSIFICATION_REPORT.exists(), f"Missing {CLASSIFICATION_REPORT}"
    assert BENCHMARK_REPORT.exists(), f"Missing {BENCHMARK_REPORT}"
    assert CONFUSION_MATRIX_IMG.exists(), f"Missing {CONFUSION_MATRIX_IMG}"
    assert TRAINING_CURVES_IMG.exists(), f"Missing {TRAINING_CURVES_IMG}"

def test_classification_report_metrics():
    """Verify classification report contains genuine metrics and valid class mappings."""
    with open(CLASSIFICATION_REPORT, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "accuracy" in data
    assert 0.0 <= data["accuracy"] <= 1.0
    assert "macro_metrics" in data
    assert "precision" in data["macro_metrics"]
    assert "recall" in data["macro_metrics"]
    assert "f1_score" in data["macro_metrics"]

    assert "classes" in data
    assert len(data["classes"]) == 4
    for cls_name in ["Apple_Bad", "Apple_Good", "Banana_Bad", "Banana_Good"]:
        assert cls_name in data["per_class_metrics"]
        assert "precision" in data["per_class_metrics"][cls_name]
        assert "recall" in data["per_class_metrics"][cls_name]
        assert "f1_score" in data["per_class_metrics"][cls_name]

    # Confusion matrix
    assert "confusion_matrix" in data
    assert len(data["confusion_matrix"]["matrix"]) == 4

def test_benchmark_report_metrics():
    """Verify latency benchmarks are non-zero empirical measurements."""
    with open(BENCHMARK_REPORT, "r", encoding="utf-8") as f:
        bench = json.load(f)

    assert "latency_ms" in bench
    assert bench["latency_ms"]["mean"] > 0.0
    assert bench["latency_ms"]["median_p50"] > 0.0
    assert bench["latency_ms"]["p95"] > 0.0
    assert bench["throughput_fps"] > 0.0

