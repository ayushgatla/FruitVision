import io
import os
from backend.database.models import FruitInspection
from backend.utils.helpers import sanitize_csv_cell

# ==========================================
# 1. API Key Authentication (HIGH-02)
# ==========================================

def test_delete_history_without_api_key_rejected(client):
    """Unauthenticated DELETE /api/history must return 401 Unauthorized."""
    res = client.delete("/api/history")
    assert res.status_code == 401
    assert "Missing administrative API key" in res.json()["detail"]
    assert "WWW-Authenticate" in res.headers

def test_delete_single_record_without_api_key_rejected(client):
    """Unauthenticated DELETE /api/history/{id} must return 401 Unauthorized."""
    res = client.delete("/api/history/1")
    assert res.status_code == 401
    assert "Missing administrative API key" in res.json()["detail"]

def test_delete_history_with_invalid_api_key_rejected(client):
    """DELETE /api/history with invalid X-API-Key must return 403 Forbidden."""
    res = client.delete("/api/history", headers={"X-API-Key": "wrong-secret-key"})
    assert res.status_code == 403
    assert "Invalid administrative API key" in res.json()["detail"]

def test_delete_history_with_valid_api_key_allowed(client, test_admin_key, db_session):
    """DELETE /api/history with valid X-API-Key must succeed (200)."""
    # Seed a record into test DB
    dummy = FruitInspection(
        image_path="/static/uploads/test.jpg",
        predicted_class="Good",
        confidence=0.95,
        inspection_status="ACCEPTED",
        fruit_count=1,
        source_type="test"
    )
    db_session.add(dummy)
    db_session.commit()

    res = client.delete("/api/history", headers={"X-API-Key": test_admin_key})
    assert res.status_code == 200
    assert res.json()["status"] == "success"
    assert res.json()["deleted_count"] >= 1

def test_delete_history_with_valid_x_admin_key_header(client, test_admin_key):
    """DELETE /api/history also accepts X-Admin-Key header."""
    res = client.delete("/api/history", headers={"X-Admin-Key": test_admin_key})
    assert res.status_code == 200
    assert res.json()["status"] == "success"

def test_reload_model_without_api_key_rejected(client):
    """Unauthenticated POST /api/reload-model must return 401 Unauthorized."""
    res = client.post("/api/reload-model")
    assert res.status_code == 401

def test_reload_model_with_invalid_api_key_rejected(client):
    """POST /api/reload-model with wrong key must return 403 Forbidden."""
    res = client.post("/api/reload-model", headers={"X-API-Key": "incorrect-key"})
    assert res.status_code == 403

def test_reload_model_with_valid_api_key_allowed(client, test_admin_key):
    """POST /api/reload-model with valid key must succeed (200)."""
    res = client.post("/api/reload-model", headers={"X-API-Key": test_admin_key})
    assert res.status_code == 200
    assert "status" in res.json()

# ==========================================
# 2. File Upload Traversal & DoS (CRIT-04, MED-03)
# ==========================================

def test_upload_path_traversal_filename_neutralized(client, dummy_jpeg_bytes):
    """Directory traversal sequences in filename must not escape uploads directory."""
    malicious_filename = "../../../../evil_test.jpg"
    files = {"file": (malicious_filename, io.BytesIO(dummy_jpeg_bytes), "image/jpeg")}
    res = client.post("/api/inspect/image", files=files)
    assert res.status_code == 200

    # Ensure no file was created in root or escaped paths
    assert not os.path.exists("evil_test.jpg")

    # Verify saved image is strictly inside the sandboxed uploads dir
    json_resp = res.json()
    saved_url = json_resp["image_url"]
    saved_basename = os.path.basename(saved_url)
    expected_path = os.path.join(client.uploads_dir, saved_basename)
    assert os.path.exists(expected_path)
    canonical_upload = os.path.abspath(client.uploads_dir)
    assert os.path.commonpath([os.path.abspath(expected_path), canonical_upload]) == canonical_upload

def test_upload_forbidden_script_extension_rejected(client):
    """Uploading non-whitelisted extensions (.py, .exe, .sh, .bat) must return 400."""
    files = {"file": ("malicious.py", io.BytesIO(b"print('hacked')"), "text/x-python")}
    res = client.post("/api/inspect/image", files=files)
    assert res.status_code == 400
    assert "Unsupported file extension" in res.json()["detail"]

def test_upload_forbidden_executable_extension_rejected(client):
    """Executable extensions like .exe must return 400 Bad Request."""
    files = {"file": ("payload.exe", io.BytesIO(b"MZ\x90\x00"), "application/x-msdownload")}
    res = client.post("/api/inspect/image", files=files)
    assert res.status_code == 400
    assert "Unsupported file extension" in res.json()["detail"]

def test_upload_corrupted_image_rejected(client):
    """Uploading corrupt or invalid bytes with .jpg extension must return 400."""
    corrupted_bytes = b"CORRUPTED_BINARY_NOT_AN_IMAGE_PAYLOAD"
    files = {"file": ("fake.jpg", io.BytesIO(corrupted_bytes), "image/jpeg")}
    res = client.post("/api/inspect/image", files=files)
    assert res.status_code == 400
    assert "Invalid image file format" in res.json()["detail"]

def test_upload_size_limit_exceeded_rejected(client):
    """Uploading payloads exceeding 10MB must return 413 Request Entity Too Large."""
    oversized_bytes = b"0" * (10 * 1024 * 1024 + 1024)  # 10MB + 1KB
    files = {"file": ("huge_image.jpg", io.BytesIO(oversized_bytes), "image/jpeg")}
    res = client.post("/api/inspect/image", files=files)
    assert res.status_code == 413
    assert "maximum allowed size" in res.json()["detail"]

# ==========================================
# 3. CORS Configuration (HIGH-04)
# ==========================================

def test_cors_allowed_origin_preflight(client):
    """Verify CORS preflight allows whitelisted frontend origin."""
    headers = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "X-API-Key,Content-Type"
    }
    res = client.options("/api/inspect/image", headers=headers)
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert res.headers.get("access-control-allow-credentials") == "true"

def test_cors_disallowed_origin_rejected(client):
    """Verify CORS preflight does not grant credentials to unauthorized origins or wildcard."""
    headers = {
        "Origin": "http://unauthorized-attacker.example.com",
        "Access-Control-Request-Method": "POST"
    }
    res = client.options("/api/inspect/image", headers=headers)
    origin_header = res.headers.get("access-control-allow-origin")
    assert origin_header != "*"
    assert origin_header != "http://unauthorized-attacker.example.com"

# ==========================================
# 4. CSV Formula Injection Escaping (HIGH-06)
# ==========================================

def test_csv_formula_injection_sanitization():
    """Verify that spreadsheet formula triggers (=, +, -, @, \\t, \\r) are escaped with single quote via backend helper."""
    assert sanitize_csv_cell("=CMD|'calc.exe'!A0") == "\"'=CMD|'calc.exe'!A0\""
    assert sanitize_csv_cell("+12345") == "\"'+12345\""
    assert sanitize_csv_cell("-SUM(A1:A10)") == "\"'-SUM(A1:A10)\""
    assert sanitize_csv_cell("@alert(1)") == "\"'@alert(1)\""
    assert sanitize_csv_cell("   =formula") == "\"'   =formula\""
    assert sanitize_csv_cell('Normal "quoted" text') == '"Normal ""quoted"" text"'
    assert sanitize_csv_cell("Standard Good Apple") == '"Standard Good Apple"'
    assert sanitize_csv_cell(None) == '""'
