import io
import os
from backend.database.models import FruitInspection

def test_inspect_single_image(client, dummy_jpeg_bytes, db_session):
    """Test standard single fruit image inspection."""
    files = {"file": ("test_fruit.jpg", io.BytesIO(dummy_jpeg_bytes), "image/jpeg")}
    data = {"notes": "Regression test single inspection"}
    res = client.post("/api/inspect/image", files=files, data=data)
    assert res.status_code == 200
    body = res.json()
    assert "predicted_class" in body
    assert body["predicted_class"] in ["Good", "Damaged", "Rotten", "Not Fruits"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert "status" in body
    assert "suggested_action" in body
    assert "probabilities" in body
    assert body["image_url"].startswith("/static/uploads/")

    # Verify DB persistence
    record = db_session.query(FruitInspection).filter(FruitInspection.id == body["id"]).first()
    assert record is not None
    assert record.notes == "Regression test single inspection"

def test_inspect_multiple_fruits(client, dummy_jpeg_bytes, db_session):
    """Test multi-fruit image inspection endpoint."""
    files = {"file": ("test_multi.png", io.BytesIO(dummy_jpeg_bytes), "image/png")}
    res = client.post("/api/inspect/multi", files=files)
    assert res.status_code == 200
    body = res.json()
    assert "total_fruits" in body
    assert "good_count" in body
    assert "damaged_count" in body
    assert "rotten_count" in body
    assert "acceptance_rate" in body
    assert "detections" in body
    assert body["image_url"].startswith("/static/uploads/")

def test_inspect_live_frame_defect_alert_saves_evidence(client, dummy_base64_frame, db_session, monkeypatch):
    """
    Test live frame inspection when defect is flagged.
    Verifies CRIT-05: draw_single_annotation and save_evidence_snapshot execute without NameError,
    and evidence snapshot is generated on disk and persisted to DB.
    """
    from backend.services.detector import get_detector
    detector = get_detector()

    # Mock predict_single_crop to simulate a high-confidence defect alert
    def mock_predict_crop(img):
        return {
            "label": "Damaged",
            "confidence": 0.92,
            "probabilities": {"Good": 0.05, "Damaged": 0.92, "Rotten": 0.03},
            "status": "DEFECT_FOUND",
            "suggested_action": "Divert to secondary trimming station"
        }

    monkeypatch.setattr(detector, "predict_single_crop", mock_predict_crop)

    payload = {
        "frame_data": dummy_base64_frame,
        "source_id": "test_webcam"
    }
    res = client.post("/api/inspect/frame", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["alert"] is True
    assert body["predicted_class"] == "Damaged"
    assert body["confidence"] == 0.92

    # Verify evidence snapshot was created in SQLite without NameError
    alert_record = db_session.query(FruitInspection).filter(FruitInspection.source_type == "webcam").first()
    assert alert_record is not None
    assert alert_record.predicted_class == "Damaged"
    assert alert_record.evidence_path is not None
    assert alert_record.evidence_path.startswith("/static/evidence/")

    # Verify snapshot file was written to disk in sandboxed evidence directory
    evidence_filename = os.path.basename(alert_record.evidence_path)
    evidence_disk_path = os.path.join(client.evidence_dir, evidence_filename)
    assert os.path.exists(evidence_disk_path)
    assert os.path.getsize(evidence_disk_path) > 0

def test_inspect_live_frame_invalid_base64_returns_400(client):
    """
    Test that invalid frame base64 encoding returns HTTP 400 Bad Request,
    verifying MED-07 (not swallowed or converted to 500).
    """
    payload = {
        "frame_data": "invalid_not_base64_string",
        "source_id": "broken_cam"
    }
    res = client.post("/api/inspect/frame", json=payload)
    assert res.status_code == 400
    assert "Invalid frame" in res.json()["detail"]

