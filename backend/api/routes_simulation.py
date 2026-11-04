import os
import cv2
import random
import shutil
from datetime import datetime
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Dict, Any, List

from backend.database.database import get_db
from backend.database.models import ConveyorEvent, FruitInspection
from backend.services.detector import get_detector
from backend.utils.helpers import SAMPLES_DIR

router = APIRouter()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMPLE_IMAGES_DIR = os.path.join(PROJECT_ROOT, "sample_images")
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# In-memory session stats
simulation_counters = {
    "total_sorted": 0,
    "good_count": 0,
    "damaged_count": 0,
    "rotten_count": 0,
    "current_speed": 1.0,
    "start_time": datetime.utcnow().isoformat(),
    "cycle_index": 0
}

def discover_sample_images() -> List[str]:
    """
    Dynamically scans sample_images directory on every invocation.
    Guarantees any newly dropped images are immediately included.
    """
    candidates = []
    if os.path.exists(SAMPLE_IMAGES_DIR):
        for f in os.listdir(SAMPLE_IMAGES_DIR):
            ext = os.path.splitext(f)[1].lower()
            if ext in VALID_IMAGE_EXTS:
                candidates.append(f)
    
    # Fallback to static/samples if sample_images is empty
    if not candidates and os.path.exists(SAMPLES_DIR):
        for f in os.listdir(SAMPLES_DIR):
            ext = os.path.splitext(f)[1].lower()
            if ext in VALID_IMAGE_EXTS:
                candidates.append(f)

    candidates.sort()
    return candidates

def infer_fruit_type(filename: str) -> str:
    """Infers display label from filename or default to Fruit."""
    fn = filename.lower()
    if "apple" in fn:
        return "Apple"
    elif "banana" in fn:
        return "Banana"
    elif "orange" in fn or "citrus" in fn:
        return "Orange"
    elif "lemon" in fn:
        return "Lemon"
    elif "pear" in fn:
        return "Pear"
    elif "not_fruit" in fn or "phone" in fn or "car" in fn or "object" in fn:
        return "Object"
    elif "multi" in fn:
        return "Batch"
    return "Fruit"

@router.get("/samples-list")
def list_available_samples():
    """
    Returns the real-time list of all image files dynamically discovered
    in the sample_images folder.
    """
    files = discover_sample_images()
    return {
        "count": len(files),
        "directory": SAMPLE_IMAGES_DIR,
        "samples": files
    }

@router.get("/sample")
def get_simulation_fruit(
    fruit_choice: str = Query("random"),
    db: Session = Depends(get_db)
):
    """
    Feeds the next fruit item on the conveyor belt,
    runs actual EfficientNetB0 CNN inference,
    and returns sorting decision and gate routing.
    Dynamically uses ALL images found in sample_images (including newly added ones).
    """
    global simulation_counters

    files = discover_sample_images()
    if not files:
        files = ["sample_good_apple.jpg"]

    # Selection logic
    if fruit_choice in ["cycle", "sequential"]:
        # Cycle through every image sequentially so every file is scanned in order
        idx = simulation_counters.get("cycle_index", 0) % len(files)
        chosen_filename = files[idx]
        simulation_counters["cycle_index"] = (idx + 1) % len(files)
    elif fruit_choice != "random" and any(fruit_choice.lower() in f.lower() for f in files):
        matched = [f for f in files if fruit_choice.lower() in f.lower()]
        chosen_filename = random.choice(matched)
    else:
        # Default: Pick across ALL available images in sample_images
        chosen_filename = random.choice(files)

    fruit_type = infer_fruit_type(chosen_filename)

    # Resolve image path
    primary_path = os.path.join(SAMPLE_IMAGES_DIR, chosen_filename)
    static_dest = os.path.join(SAMPLES_DIR, chosen_filename)

    if os.path.exists(primary_path):
        sample_path = primary_path
        # Sync to static/samples as backup
        if not os.path.exists(static_dest):
            try:
                shutil.copy2(primary_path, static_dest)
            except Exception:
                pass
    elif os.path.exists(static_dest):
        sample_path = static_dest
    else:
        sample_path = primary_path

    img_bgr = None
    if os.path.exists(sample_path):
        img_bgr = cv2.imread(sample_path)

    detector = get_detector()
    if img_bgr is not None:
        pred = detector.predict_single_crop(img_bgr)
    else:
        pred = {
            "label": "Good",
            "confidence": 0.95,
            "probabilities": {"Good": 0.95, "Damaged": 0.03, "Rotten": 0.02},
            "status": "ACCEPTED",
            "suggested_action": "Direct to Automated Packaging Line"
        }

    label = pred["label"]
    conf = pred["confidence"]

    # Determine Industrial Sorting Gate
    if label == "Good":
        sorting_gate = "GATE_1_ACCEPTED"
        action = "Accepted: Divert to Primary Retail Packaging Line"
        simulation_counters["good_count"] += 1
    elif label == "Damaged":
        sorting_gate = "GATE_2_SECONDARY"
        action = "Secondary: Route to Manual Trimming & Puree Station"
        simulation_counters["damaged_count"] += 1
    elif label in ["Not Fruits", "Not Fruit"]:
        sorting_gate = "GATE_3_REJECTED"
        action = "Rejected: Foreign Non-Produce Ejected to Diverter Chute"
        simulation_counters["rotten_count"] += 1
    else:  # Rotten
        sorting_gate = "GATE_3_REJECTED"
        action = "Rejected: Pneumatic Ejection to Organic Bio-Waste Bin"
        simulation_counters["rotten_count"] += 1

    simulation_counters["total_sorted"] += 1

    # Unique tracking item ID
    item_id = f"FRUIT-{datetime.utcnow().strftime('%H%M%S')}-{random.randint(100, 999)}"

    # Save conveyor event to DB
    event = ConveyorEvent(
        item_id=item_id,
        fruit_type=fruit_type,
        predicted_quality=label,
        confidence=conf,
        sorting_action=action,
        gate_routed=sorting_gate
    )
    db.add(event)

    # Also log to main inspection audit
    inspection = FruitInspection(
        image_path=f"/sample_images/{chosen_filename}",
        predicted_class=label,
        confidence=conf,
        inspection_status=sorting_gate,
        fruit_count=1,
        good_count=1 if label == "Good" else 0,
        damaged_count=1 if label == "Damaged" else 0,
        rotten_count=1 if label == "Rotten" else 0,
        source_type="conveyor_simulation",
        notes=f"Simulation item {item_id} ({chosen_filename})"
    )
    db.add(inspection)
    db.commit()

    return {
        "item_id": item_id,
        "fruit_type": fruit_type,
        "sample_filename": chosen_filename,
        "predicted_class": label,
        "confidence": conf,
        "probabilities": pred["probabilities"],
        "sorting_gate": sorting_gate,
        "action": action,
        "image_url": f"/sample_images/{chosen_filename}",
        "timestamp": datetime.utcnow().isoformat(),
        "live_counters": simulation_counters,
        "available_samples_count": len(files)
    }

@router.get("/stats")
def get_simulation_stats():
    """Retrieve live conveyor simulation counters."""
    total = simulation_counters["total_sorted"]
    good = simulation_counters["good_count"]
    damaged = simulation_counters["damaged_count"]
    rotten = simulation_counters["rotten_count"]
    acceptance = round((good / total * 100), 1) if total > 0 else 100.0

    return {
        "total_sorted": total,
        "good_count": good,
        "damaged_count": damaged,
        "rotten_count": rotten,
        "acceptance_rate": acceptance,
        "speed": simulation_counters["current_speed"]
    }

@router.post("/reset")
def reset_simulation_stats():
    """Reset the conveyor belt simulation counters."""
    global simulation_counters
    simulation_counters = {
        "total_sorted": 0,
        "good_count": 0,
        "damaged_count": 0,
        "rotten_count": 0,
        "current_speed": 1.0,
        "start_time": datetime.utcnow().isoformat(),
        "cycle_index": 0
    }
    return {"status": "success", "message": "Simulation counters reset."}
