import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.database.database import init_db
from backend.services.detector import get_detector
from backend.utils.helpers import ensure_directories, STATIC_DIR
from backend.utils.auth import verify_admin_key
from backend.api import api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("[FruitVision AI] Initializing storage directories and SQLite database...")
    ensure_directories()
    init_db()

    print("[FruitVision AI] Loading EfficientNetB0 fruit quality classifier...")
    detector = get_detector()
    print(f"[FruitVision AI] Ready. Device: {detector.device}. Custom weights active: {detector.using_custom_weights}")

    yield
    # Shutdown
    print("[FruitVision AI] Shutting down FruitVision AI services.")

app = FastAPI(
    title="FruitVision AI — Intelligent Fruit Quality Inspection & Sorting System",
    description="Industry 4.0 Computer Vision backend for automated fruit grading, quality classification, and conveyor sorting.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS security configuration (HIGH-04)
DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

env_cors = os.getenv("ALLOWED_ORIGINS") or os.getenv("FRUITVISION_CORS_ORIGINS")
if env_cors and env_cors.strip():
    allowed_origins = [origin.strip() for origin in env_cors.split(",") if origin.strip()]
else:
    allowed_origins = DEFAULT_ALLOWED_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Admin-Key", "Accept"],
    max_age=600,
)

# Mount static files
ensure_directories()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Include API routes
app.include_router(api_router)

@app.get("/")
def root():
    return {
        "system": "FruitVision AI — Intelligent Fruit Quality Inspection & Sorting System",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "online"
    }

@app.get("/api/health")
def health_check():
    detector = get_detector()
    return {
        "status": "healthy",
        "device": str(detector.device).upper(),
        "model_loaded": detector.model is not None,
        "model_path": detector.weights_path or "ImageNet Pretrained Backbone",
        "using_custom_weights": detector.using_custom_weights,
        "classes": detector.classes,
        "db_status": "connected",
        "version": "1.0.0"
    }

@app.post("/api/reload-model", dependencies=[Depends(verify_admin_key)])
def reload_model():
    """Reload custom model weights (Admin authenticated)."""
    detector = get_detector()
    reloaded = detector.reload_custom_weights()
    return {
        "status": "success" if reloaded else "no_custom_weights",
        "using_custom_weights": detector.using_custom_weights,
        "model_path": detector.weights_path
    }

if __name__ == "__main__":
    import uvicorn
    import sys

    port = 8001
    if "--port" in sys.argv:
        p_idx = sys.argv.index("--port") + 1
        if p_idx < len(sys.argv):
            port = int(sys.argv[p_idx])

    print(f"[FruitVision AI] Starting server on http://127.0.0.1:{port}...")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)
