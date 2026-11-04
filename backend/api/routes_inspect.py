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
from backend.utils.helpers import UPLOADS_DIR, base64_to_numpy, ensure_directories
from backend.models.schemas import SingleInspectionResponse, MultiFruitInspectionResponse

router = APIRouter()

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
    """Inspect and classify a single fruit image using EfficientNetB0."""
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
    upload_filename = f"upload_{ts}_{file.filename}"
    upload_path = os.path.join(UPLOADS_DIR, upload_filename)
    with open(upload_path, "wb") as f:
        f.write(contents)
    upload_url = f"/static/uploads/{upload_filename}"

    # Perform detection and classification
    result = detector.detect_and_classify_image(img_bgr, source_id=file.filename)

    # Save record to SQLite database
    inspection = FruitInspection(
        image_path=upload_url,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        inspection_status=result["status"],
        fruit_count=1,
        good_count=1 if result["predicted_class"] == "Good" else 0,
        damaged_count=1 if result["predicted_class"] == "Damaged" else 0,
        rotten_count=1 if result["predicted_class"] == "Rotten" else 0,
        source_type="single_upload",
        notes=notes or f"Inspection of {file.filename}",
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
        "model_version": result["model_version"]
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
    upload_filename = f"multi_{ts}_{file.filename}"
    upload_path = os.path.join(UPLOADS_DIR, upload_filename)
    with open(upload_path, "wb") as f:
        f.write(contents)
    upload_url = f"/static/uploads/{upload_filename}"

    # Perform multi-fruit inspection
    multi_res = detector.detect_multiple_fruits(img_bgr)

    # Save to SQLite database
    primary_pred = "Good"
    if multi_res["rotten_count"] > 0:
        primary_pred = "Rotten"
    elif multi_res["damaged_count"] > 0:
        primary_pred = "Damaged"

    inspection = FruitInspection(
        image_path=upload_url,
        predicted_class=primary_pred,
        confidence=round(multi_res["acceptance_rate"] / 100.0, 2),
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
        "detections": multi_res["detections"],
        "image_url": upload_url,
        "annotated_image_url": multi_res["annotated_image_url"],
        "timestamp": inspection.inspection_timestamp.isoformat()
    }

@router.post("/frame")
async def inspect_live_frame(payload: FramePayload, db: Session = Depends(get_db)):
    """Process live video stream or webcam frame with EfficientNetB0 HUD."""
    try:
        img_bgr = base64_to_numpy(payload.frame_data)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid frame base64 encoding.")

        detector = get_detector()
        # Single crop or multi inspection depending on frame composition
        pred = detector.predict_single_crop(img_bgr)

        # Log defect alert to database if confidence is high (> 80%)
        if pred["label"] in ["Damaged", "Rotten"] and pred["confidence"] >= 0.80:
            evidence_url = None
            try:
                annotated = draw_single_annotation(img_bgr, pred["label"], pred["confidence"])
                _, evidence_url = save_evidence_snapshot(annotated, prefix="live_alert")
            except Exception:
                pass

            inspection = FruitInspection(
                image_path=evidence_url,
                predicted_class=pred["label"],
                confidence=pred["confidence"],
                inspection_status=pred["status"],
                fruit_count=1,
                good_count=0,
                damaged_count=1 if pred["label"] == "Damaged" else 0,
                rotten_count=1 if pred["label"] == "Rotten" else 0,
                source_type="webcam",
                notes=f"Live stream alert: {pred['label']}",
                evidence_path=evidence_url
            )
            db.add(inspection)
            db.commit()

        return {
            "predicted_class": pred["label"],
            "confidence": pred["confidence"],
            "status": pred["status"],
            "suggested_action": pred["suggested_action"],
            "probabilities": pred["probabilities"],
            "alert": pred["label"] in ["Damaged", "Rotten"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

