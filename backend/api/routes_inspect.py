import os
import cv2
import numpy as np
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional
from pydantic import BaseModel

from backend.database.database import get_db
from backend.database.models import FruitInspection
from backend.services.detector import get_detector
from backend.utils.helpers import (
    UPLOADS_DIR,
    base64_to_numpy,
    ensure_directories,
    draw_single_annotation,
    draw_multi_annotations,
    save_evidence_snapshot
)
from backend.models.schemas import SingleInspectionResponse, MultiFruitInspectionResponse

router = APIRouter()

# Global alert debounce timer for webcam live stream
_last_live_alert_time = datetime.min

class FramePayload(BaseModel):
    frame_data: str
    source_id: Optional[str] = "webcam"

@router.post("/image", response_model=SingleInspectionResponse)
@router.post("/detect/image", response_model=SingleInspectionResponse)
async def inspect_single_image(
    file: UploadFile = File(...),
    notes: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Inspect and classify a fruit image using EfficientNetB0.
    Automatically detects if multiple fruits exist on screen, identifies each with bounding boxes,
    classifies each separately & together, and records the audit record.
    """
    ensure_directories()
    detector = get_detector()

    # Read uploaded file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file format.")

    # Save original upload
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    clean_filename = os.path.basename(file.filename)
    upload_filename = f"upload_{ts}_{clean_filename}"
    upload_path = os.path.join(UPLOADS_DIR, upload_filename)
    with open(upload_path, "wb") as f:
        f.write(contents)
    upload_url = f"/static/uploads/{upload_filename}"

    # Perform detection and classification
    result = detector.detect_and_classify_image(img_bgr, source_id=clean_filename)

    is_multi = result.get("is_multi", False)
    total_fruits = result.get("total_fruits", 1)
    good_count = result.get("good_count", 1 if result["predicted_class"] == "Good" else 0)
    damaged_count = result.get("damaged_count", 1 if result["predicted_class"] == "Damaged" else 0)
    rotten_count = result.get("rotten_count", 1 if result["predicted_class"] == "Rotten" else 0)

    default_note = (
        f"Batch inspection: {total_fruits} fruits identified (Good: {good_count}, Damaged: {damaged_count}, Rotten: {rotten_count})"
        if is_multi else f"Single inspection of {file.filename}"
    )

    # Save record to SQLite database
    inspection = FruitInspection(
        image_path=upload_url,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        inspection_status=result["status"],
        fruit_count=total_fruits,
        good_count=good_count,
        damaged_count=damaged_count,
        rotten_count=rotten_count,
        source_type="multi_upload" if is_multi else "single_upload",
        notes=notes or default_note,
        evidence_path=result["annotated_image_url"]
    )
    db.add(inspection)
    db.commit()
    db.refresh(inspection)

    return {
        "id": inspection.id,
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "status": result["status"],
        "suggested_action": result["suggested_action"],
        "probabilities": result["probabilities"],
        "image_url": upload_url,
        "annotated_image_url": result["annotated_image_url"],
        "timestamp": inspection.inspection_timestamp.isoformat(),
        "device": result["device"],
        "model_version": result["model_version"],
        "is_multi": is_multi,
        "fruit_count": total_fruits,
        "good_count": good_count,
        "damaged_count": damaged_count,
        "rotten_count": rotten_count,
        "acceptance_rate": result.get("acceptance_rate", 100.0),
        "detections": result.get("detections", [])
    }

@router.post("/multi", response_model=MultiFruitInspectionResponse)
async def inspect_multiple_fruits(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Detect and classify multiple fruits in a single image."""
    ensure_directories()
    detector = get_detector()

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file format.")

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    clean_filename = os.path.basename(file.filename)
    upload_filename = f"multi_{ts}_{clean_filename}"
    upload_path = os.path.join(UPLOADS_DIR, upload_filename)
    with open(upload_path, "wb") as f:
        f.write(contents)
    upload_url = f"/static/uploads/{upload_filename}"

    # Perform multi-fruit inspection
    multi_res = detector.detect_multiple_fruits(img_bgr)

    # Save to SQLite database
    inspection = FruitInspection(
        image_path=upload_url,
        predicted_class=multi_res["predicted_class"],
        confidence=multi_res["confidence"],
        inspection_status=multi_res["overall_status"],
        fruit_count=multi_res["total_fruits"],
        good_count=multi_res["good_count"],
        damaged_count=multi_res["damaged_count"],
        rotten_count=multi_res["rotten_count"],
        source_type="multi_upload",
        notes=f"Batch of {multi_res['total_fruits']} fruits. Acceptance: {multi_res['acceptance_rate']}%",
        evidence_path=multi_res["annotated_image_url"]
    )
    db.add(inspection)
    db.commit()
    db.refresh(inspection)

    return {
        "id": inspection.id,
        "total_fruits": multi_res["total_fruits"],
        "good_count": multi_res["good_count"],
        "damaged_count": multi_res["damaged_count"],
        "rotten_count": multi_res["rotten_count"],
        "acceptance_rate": multi_res["acceptance_rate"],
        "overall_status": multi_res["overall_status"],
        "predicted_class": multi_res["predicted_class"],
        "confidence": multi_res["confidence"],
        "probabilities": multi_res["probabilities"],
        "detections": multi_res["detections"],
        "image_url": upload_url,
        "annotated_image_url": multi_res["annotated_image_url"],
        "timestamp": inspection.inspection_timestamp.isoformat()
    }

@router.post("/frame")
async def inspect_live_frame(payload: FramePayload, db: Session = Depends(get_db)):
    """
    Process live video stream or webcam frame with real-time fruit instance segmentation,
    individual bounding box tracking, and defect alerts.
    Accurately suppresses human faces/skin, room clutter, and non-fruit items.
    """
    global _last_live_alert_time
    try:
        img_bgr = base64_to_numpy(payload.frame_data)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid frame base64 encoding.")

        detector = get_detector()
        h, w = img_bgr.shape[:2]

        # Use specialized webcam saliency detector (suppresses faces, skin, room walls)
        boxes = detector.find_fruit_boxes_live(img_bgr)

        # 1. No genuine fruit candidate detected in frame (e.g. user sitting at desk / background)
        if len(boxes) == 0:
            return {
                "predicted_class": "Not Fruits",
                "confidence": 0.99,
                "status": "NON_PRODUCE",
                "suggested_action": "Optical sensor active: Place fruit in camera view to begin inspection.",
                "probabilities": {
                    "Not Fruits": 0.99,
                    "Good": 0.0,
                    "Damaged": 0.0,
                    "Rotten": 0.01
                },
                "alert": False,
                "total_fruits": 0,
                "good_count": 0,
                "damaged_count": 0,
                "rotten_count": 0,
                "acceptance_rate": 100.0,
                "detections": [],
                "evidence_url": None
            }

        # 2. One or more candidates localized
        detections = []
        for (x1, y1, x2, y2) in boxes:
            crop = img_bgr[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
                crop = img_bgr

            pred = detector.predict_single_crop(crop)
            detections.append({
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "label": pred["label"],
                "confidence": pred["confidence"],
                "status": pred["status"],
                "probabilities": pred.get("probabilities", {})
            })

        # Separate genuine produce instances from non-produce items
        fruit_dets = [d for d in detections if d["label"] in ["Good", "Damaged", "Rotten"]]

        if len(fruit_dets) == 0:
            # All candidates were identified as non-produce objects (e.g. phone, cup, paper)
            top_det = detections[0]
            return {
                "predicted_class": "Not Fruits",
                "confidence": top_det["confidence"],
                "status": "NON_PRODUCE",
                "suggested_action": "Foreign non-produce object identified. Position fresh fruit in view.",
                "probabilities": {
                    "Not Fruits": top_det["confidence"],
                    "Good": 0.0,
                    "Damaged": 0.0,
                    "Rotten": round(1.0 - top_det["confidence"], 4)
                },
                "alert": False,
                "total_fruits": 0,
                "good_count": 0,
                "damaged_count": 0,
                "rotten_count": 0,
                "acceptance_rate": 100.0,
                "detections": detections,
                "evidence_url": None
            }

        # Valid fruit(s) on screen
        total_fruits = len(fruit_dets)
        good_count = sum(1 for d in fruit_dets if d["label"] == "Good")
        damaged_count = sum(1 for d in fruit_dets if d["label"] == "Damaged")
        rotten_count = sum(1 for d in fruit_dets if d["label"] == "Rotten")
        acceptance_rate = round((good_count / total_fruits) * 100, 1)

        if rotten_count > 0:
            predicted_class = "Rotten"
            status = "REJECTED"
            suggested_action = "Immediate Diverter Ejection to Bio-Waste / Composting"
            rotten_confs = [d["confidence"] for d in fruit_dets if d["label"] == "Rotten"]
            confidence = max(rotten_confs) if rotten_confs else 0.95
            alert = True
        elif damaged_count > 0:
            predicted_class = "Damaged"
            status = "SECONDARY_INSPECTION"
            suggested_action = "Route to Secondary Manual Grading / Reprocessing Unit"
            damaged_confs = [d["confidence"] for d in fruit_dets if d["label"] == "Damaged"]
            confidence = max(damaged_confs) if damaged_confs else 0.90
            alert = True
        else:
            predicted_class = "Good"
            status = "ACCEPTED"
            suggested_action = "Direct to Automated Packaging Line (Market Grade A)"
            good_confs = [d["confidence"] for d in fruit_dets if d["label"] == "Good"]
            confidence = float(round(sum(good_confs) / len(good_confs), 4)) if good_confs else 0.98
            alert = False

        probabilities = {
            "Good": round(good_count / total_fruits, 4),
            "Damaged": round(damaged_count / total_fruits, 4),
            "Rotten": round(rotten_count / total_fruits, 4)
        }

        # Save annotated evidence snapshot
        if total_fruits > 1:
            annotated = draw_multi_annotations(img_bgr, fruit_dets)
        else:
            first = fruit_dets[0]
            annotated = draw_single_annotation(
                img_bgr, first["label"], first["confidence"],
                box=(first["x1"], first["y1"], first["x2"], first["y2"])
            )
        _, evidence_url = save_evidence_snapshot(annotated, prefix="live_stream")

        # Log defect alert to database with 5-second debounce to prevent log flooding
        now = datetime.utcnow()
        if alert and confidence >= 0.82 and (now - _last_live_alert_time).total_seconds() >= 5.0:
            _last_live_alert_time = now
            inspection = FruitInspection(
                image_path=evidence_url,
                predicted_class=predicted_class,
                confidence=confidence,
                inspection_status=status,
                fruit_count=total_fruits,
                good_count=good_count,
                damaged_count=damaged_count,
                rotten_count=rotten_count,
                source_type="webcam",
                notes=f"Live stream alert: {predicted_class} ({total_fruits} fruits)",
                evidence_path=evidence_url
            )
            db.add(inspection)
            db.commit()

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "status": status,
            "suggested_action": suggested_action,
            "probabilities": probabilities,
            "alert": alert,
            "total_fruits": total_fruits,
            "good_count": good_count,
            "damaged_count": damaged_count,
            "rotten_count": rotten_count,
            "acceptance_rate": acceptance_rate,
            "detections": detections,
            "evidence_url": evidence_url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
