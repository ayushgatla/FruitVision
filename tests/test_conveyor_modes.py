"""
Integration tests for FruitVision AI dedicated conveyor modes (Apple / Banana / All),
dynamic 4-class detector prediction contracts, and model metrics API.
"""

import pytest
from backend.services.detector import get_detector
import numpy as np

def test_simulation_mode_apple(client):
    """Verify Apple dedicated conveyor mode only serves Apple specimens."""
    res = client.get("/api/simulation/sample?mode=apple")
    assert res.status_code == 200
    data = res.json()
    assert data["fruit_type"] == "Apple"
    assert data["conveyor_mode"] == "apple"
    assert data["sorting_gate"] in ["GATE_1_ACCEPTED", "GATE_2_SECONDARY", "GATE_3_REJECTED"]
    assert "probabilities" in data

def test_simulation_mode_banana(client):
    """Verify Banana dedicated conveyor mode only serves Banana specimens."""
    res = client.get("/api/simulation/sample?mode=banana")
    assert res.status_code == 200
    data = res.json()
    assert data["fruit_type"] == "Banana"
    assert data["conveyor_mode"] == "banana"
    assert data["sorting_gate"] in ["GATE_1_ACCEPTED", "GATE_2_SECONDARY", "GATE_3_REJECTED"]
    assert "probabilities" in data

def test_simulation_mode_all(client):
    """Verify default all-produce mode serves valid fruit and increments counters."""
    res = client.get("/api/simulation/sample?mode=all")
    assert res.status_code == 200
    data = res.json()
    assert data["fruit_type"] in ["Apple", "Orange", "Banana"]
    assert data["sorting_gate"] in ["GATE_1_ACCEPTED", "GATE_2_SECONDARY", "GATE_3_REJECTED"]
    assert data["live_counters"]["total_sorted"] >= 1

def test_analytics_model_metrics_returns_empirical_data(client):
    """Verify model-metrics endpoint returns empirical data from held-out test report."""
    res = client.get("/api/analytics/model-metrics")
    assert res.status_code == 200
    data = res.json()
    assert "overall_test_accuracy" in data
    assert 0.0 <= data["overall_test_accuracy"] <= 100.0
    assert "confusion_matrix" in data
    assert "per_class_report" in data
    assert len(data["per_class_report"]) >= 3

def test_detector_4class_mapping_contract():
    """Verify that detector dynamically maps 4 classes to canonical quality and fruit types."""
    detector = get_detector()
    assert detector.model is not None
    assert len(detector.classes) == 4

    # Test with dummy image
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    dummy_img[:] = (20, 180, 40) # Green pigment for fruit verification
    pred = detector.predict_single_crop(dummy_img)

    assert "label" in pred
    assert pred["label"] in ["Good", "Damaged", "Rotten", "Not Fruits"]
    assert "status" in pred
    assert pred["status"] in ["ACCEPTED", "SECONDARY_INSPECTION", "REJECTED", "NON_PRODUCE"]
    assert "fruit_type" in pred
    assert "probabilities" in pred
    assert len(pred["probabilities"]) == 4

