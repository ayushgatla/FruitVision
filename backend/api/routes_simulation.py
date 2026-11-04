import os
import cv2
import random
import threading
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Dict, Any

from backend.database.database import get_db
from backend.database.models import ConveyorEvent, FruitInspection
from backend.services.detector import get_detector
from backend.utils.helpers import SAMPLES_DIR

router = APIRouter()

# Thread-safe in-memory session stats
simulation_lock = threading.Lock()
simulation_counters = {
    "total_sorted": 0,
    "good_count": 0,
    "damaged_count": 0,
    "rotten_count": 0,
    "current_speed": 1.0,
    "start_time": datetime.now(timezone.utc).isoformat()
}

SAMPLE_FILES = [
    ("sample_good_apple.jpg", "Apple"),
    ("sample_good_orange.jpg", "Orange"),
    ("sample_damaged_apple.jpg", "Apple"),
    ("sample_damaged_orange.jpg", "Orange"),
    ("sample_rotten_apple.jpg", "Apple"),
    ("sample_rotten_banana.jpg", "Banana")
]


@router.get("/sample")
def get_simulation_fruit(
    fruit_choice: str = Query("random"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Feeds the next fruit item on the conveyor belt,
    runs actual EfficientNetB0 CNN inference,
    and returns sorting decision and gate routing.
    """
    # Pick sample image
    if fruit_choice != "random" and any(fruit_choice in f[0] for f in SAMPLE_FILES):
        sample_entry = next(f for f in SAMPLE_FILES if fruit_choice in f[0])
    else:
        # Realistic industrial distribution: 65% Good, 20% Damaged, 15% Rotten
        rand_roll = random.random()
        if rand_roll < 0.65:
            candidates = [f for f in SAMPLE_FILES if "good" in f[0]]
        elif rand_roll < 0.85:
            candidates = [f for f in SAMPLE_FILES if "damaged" in f[0]]
        else:
            candidates = [f for f in SAMPLE_FILES if "rotten" in f[0]]
        sample_entry = random.choice(candidates if candidates else SAMPLE_FILES)

    filename, fruit_type = sample_entry
    sample_path = os.path.join(SAMPLES_DIR, filename)

    img_bgr = None
    if os.path.exists(sample_path):
        img_bgr = cv2.imread(sample_path)

    detector = get_detector()
    if img_bgr is not None:
        pred = detector.predict_single_crop(img_bgr)
    else:
        # Safe fallback if asset missing: flag for secondary inspection, NEVER pass as Grade A Good
        expected_label = "Rotten" if "rotten" in filename.lower() else ("Damaged" if "damaged" in filename.lower() else "Damaged")
        pred = {
            "label": expected_label,
            "confidence": 0.85,
            "probabilities": {
                "Good": 0.05,
                "Damaged": 0.85 if expected_label == "Damaged" else 0.10,
                "Rotten": 0.85 if expected_label == "Rotten" else 0.10
            },
            "status": "SECONDARY_INSPECTION" if expected_label == "Damaged" else "REJECTED",
            "suggested_action": "Manual Inspection Required - Sample Image Missing"
        }

    label = pred["label"]
    conf = pred["confidence"]

    # Determine Industrial Sorting Gate and thread-safely increment counters
    with simulation_lock:
        if label == "Good":
            sorting_gate = "GATE_1_ACCEPTED"
            action = "Accepted: Divert to Primary Retail Packaging Line"
            simulation_counters["good_count"] += 1
        elif label == "Damaged":
            sorting_gate = "GATE_2_SECONDARY"
            action = "Secondary: Route to Manual Trimming & Puree Station"
            simulation_counters["damaged_count"] += 1
        else: # Rotten
            sorting_gate = "GATE_3_REJECTED"
            action = "Rejected: Pneumatic Ejection to Organic Bio-Waste Bin"
            simulation_counters["rotten_count"] += 1

        simulation_counters["total_sorted"] += 1
        current_counters = dict(simulation_counters)

    # Unique tracking item ID
    item_id = f"FRUIT-{datetime.now(timezone.utc).strftime('%H%M%S')}-{random.randint(100, 999)}"

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
        image_path=f"/static/samples/{filename}",
        predicted_class=label,
        confidence=conf,
        inspection_status=sorting_gate,
        fruit_count=1,
        good_count=1 if label == "Good" else 0,
        damaged_count=1 if label == "Damaged" else 0,
        rotten_count=1 if label == "Rotten" else 0,
        source_type="conveyor_simulation",
        notes=f"Simulation item {item_id} ({fruit_type})"
    )
    db.add(inspection)
    db.commit()

    return {
        "item_id": item_id,
        "fruit_type": fruit_type,
        "sample_filename": filename,
        "predicted_class": label,
        "confidence": conf,
        "probabilities": pred["probabilities"],
        "sorting_gate": sorting_gate,
        "action": action,
        "image_url": f"/static/samples/{filename}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "live_counters": current_counters
    }


@router.get("/stats")
def get_simulation_stats() -> Dict[str, Any]:
    """Retrieve live conveyor simulation counters (Thread-safe)."""
    with simulation_lock:
        total = simulation_counters["total_sorted"]
        good = simulation_counters["good_count"]
        damaged = simulation_counters["damaged_count"]
        rotten = simulation_counters["rotten_count"]
        speed = simulation_counters["current_speed"]

    acceptance = round((good / total * 100), 1) if total > 0 else 100.0

    return {
        "total_sorted": total,
        "good_count": good,
        "damaged_count": damaged,
        "rotten_count": rotten,
        "acceptance_rate": acceptance,
        "speed": speed
    }


@router.post("/reset")
def reset_simulation_stats() -> Dict[str, Any]:
    """Reset the conveyor belt simulation counters (Thread-safe)."""
    with simulation_lock:
        simulation_counters["total_sorted"] = 0
        simulation_counters["good_count"] = 0
        simulation_counters["damaged_count"] = 0
        simulation_counters["rotten_count"] = 0
        simulation_counters["current_speed"] = 1.0
        simulation_counters["start_time"] = datetime.now(timezone.utc).isoformat()

    return {"status": "success", "message": "Simulation counters reset."}
