import os
import cv2
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.testclient import TestClient

# Set test environment variables BEFORE importing backend app
TEST_ADMIN_KEY = "test-secret-admin-key-2026"
os.environ["FRUITVISION_API_KEY"] = TEST_ADMIN_KEY
os.environ["FRUITVISION_ADMIN_KEY"] = TEST_ADMIN_KEY
os.environ["ALLOWED_ORIGINS"] = "http://localhost:5173,http://127.0.0.1:5173"

from backend.main import app
from backend.database.database import Base, get_db
import backend.utils.helpers as helpers

@pytest.fixture(scope="session")
def test_admin_key():
    return TEST_ADMIN_KEY

@pytest.fixture
def admin_headers():
    return {"X-API-Key": TEST_ADMIN_KEY}

@pytest.fixture
def db_session():
    """Isolated in-memory SQLite database for testing."""
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    Base.metadata.create_all(bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        Base.metadata.drop_all(bind=test_engine)

@pytest.fixture
def client(db_session, tmp_path, monkeypatch):
    """
    Isolated FastAPI TestClient with sandboxed temporary directories
    and in-memory database dependency override.
    """
    test_static = tmp_path / "static"
    test_uploads = test_static / "uploads"
    test_evidence = test_static / "evidence"
    test_uploads.mkdir(parents=True, exist_ok=True)
    test_evidence.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(helpers, "STATIC_DIR", str(test_static))
    monkeypatch.setattr(helpers, "UPLOADS_DIR", str(test_uploads))
    monkeypatch.setattr(helpers, "EVIDENCE_DIR", str(test_evidence))

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        test_client.uploads_dir = str(test_uploads)
        test_client.evidence_dir = str(test_evidence)
        yield test_client
    app.dependency_overrides.clear()

@pytest.fixture
def dummy_image_bgr():
    """Generates a synthetic 224x224 RGB fruit-like test image."""
    img = np.full((224, 224, 3), (240, 240, 240), dtype=np.uint8)
    cv2.circle(img, (112, 112), 70, (20, 20, 220), -1)  # Red circle
    return img

@pytest.fixture
def dummy_jpeg_bytes(dummy_image_bgr):
    """Encodes dummy BGR image into JPEG bytes."""
    _, buf = cv2.imencode('.jpg', dummy_image_bgr)
    return buf.tobytes()

@pytest.fixture
def dummy_base64_frame(dummy_image_bgr):
    """Encodes dummy BGR image into base64 data URI."""
    return helpers.numpy_to_base64(dummy_image_bgr)

