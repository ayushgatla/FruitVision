import os
from backend.utils.helpers import SAMPLES_DIR

def test_damaged_orange_sample_asset_exists():
    """Verify CRIT-03: sample_damaged_orange.jpg exists in static samples and sample_images."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    samples_path = os.path.join(base_dir, "backend", "static", "samples", "sample_damaged_orange.jpg")
    sample_images_path = os.path.join(base_dir, "sample_images", "sample_damaged_orange.jpg")

    assert os.path.exists(samples_path), "sample_damaged_orange.jpg missing in backend/static/samples"
    assert os.path.exists(sample_images_path), "sample_damaged_orange.jpg missing in sample_images"
    assert os.path.getsize(samples_path) > 0
    assert os.path.getsize(sample_images_path) > 0

def test_simulation_damaged_orange_sorting(client):
    """
    Test simulation with sample_damaged_orange.jpg.
    Verifies it does not falsely route to GATE_1_ACCEPTED as Good produce.
    """
    res = client.get("/api/simulation/sample?fruit_choice=sample_damaged_orange.jpg")
    assert res.status_code == 200
    body = res.json()
    assert body["sample_filename"] == "sample_damaged_orange.jpg"
    assert body["fruit_type"] == "Orange"
    assert body["predicted_class"] in ["Damaged", "Rotten"]
    assert body["sorting_gate"] in ["GATE_2_SECONDARY", "GATE_3_REJECTED"]
    assert body["sorting_gate"] != "GATE_1_ACCEPTED"

def test_simulation_missing_asset_fallback_never_passes_as_good(client, monkeypatch):
    """
    Verify fallback logic if an asset file cannot be read:
    Missing/damaged fruit must never be accepted to GATE_1_ACCEPTED.
    """
    # Force os.path.exists to return False for a damaged sample
    real_exists = os.path.exists
    def mock_exists(path):
        if "sample_damaged_apple.jpg" in str(path):
            return False
        return real_exists(path)

    monkeypatch.setattr(os.path, "exists", mock_exists)

    res = client.get("/api/simulation/sample?fruit_choice=sample_damaged_apple.jpg")
    assert res.status_code == 200
    body = res.json()
    assert body["predicted_class"] != "Good"
    assert body["sorting_gate"] != "GATE_1_ACCEPTED"

def test_simulation_stats_and_reset(client):
    """Test simulation counters query and reset."""
    # Step simulation
    client.get("/api/simulation/sample")
    stats_res = client.get("/api/simulation/stats")
    assert stats_res.status_code == 200
    stats = stats_res.json()
    assert stats["total_sorted"] >= 1

    # Reset
    reset_res = client.post("/api/simulation/reset")
    assert reset_res.status_code == 200
    assert reset_res.json()["status"] == "success"

    stats_after = client.get("/api/simulation/stats").json()
    assert stats_after["total_sorted"] == 0

