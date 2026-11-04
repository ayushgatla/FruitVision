import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.database.database import init_db
from backend.services.detector import get_detector
from backend.utils.helpers import ensure_directories, STATIC_DIR
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

# CORS middleware for local Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

@app.post("/api/reload-model")
def reload_model():
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

