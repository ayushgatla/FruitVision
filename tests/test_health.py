def test_root_endpoint(client):
    """Test API root metadata endpoint."""
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "FruitVision AI" in data["system"]
    assert "version" in data

def test_health_check_endpoint(client):
    """Test /api/health returns 200 with healthy status and model parameters."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["db_status"] == "connected"
    assert isinstance(data["classes"], list)
    assert len(data["classes"]) >= 3
