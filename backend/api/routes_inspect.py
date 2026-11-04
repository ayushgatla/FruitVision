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
    upload_filename = f"upload_{ts}_{file.filename}"
    upload_path = os.path.join(UPLOADS_DIR, upload_filename)
    with open(upload_path, "wb") as f:
        f.write(contents)
    upload_url = f"/static/uploads/{upload_filename}"

    # Perform detection and classification
    result = detector.detect_and_classify_image(img_bgr, source_id=file.filename)

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
    upload_filename = f"multi_{ts}_{file.filename}"
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
    """
    try:
        img_bgr = base64_to_numpy(payload.frame_data)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid frame base64 encoding.")

        detector = get_detector()
        h, w = img_bgr.shape[:2]
        boxes = detector.find_fruit_boxes(img_bgr)

        # Multi-fruit vs Single-fruit vs Non-produce logic
        if len(boxes) > 1:
            multi_res = detector.detect_multiple_fruits(img_bgr)
            detections = multi_res["detections"]
            total_fruits = multi_res["total_fruits"]
            good_count = multi_res["good_count"]
            damaged_count = multi_res["damaged_count"]
            rotten_count = multi_res["rotten_count"]
            acceptance_rate = multi_res["acceptance_rate"]
            predicted_class = multi_res["predicted_class"]
            confidence = multi_res["confidence"]
            status = multi_res["overall_status"]
            suggested_action = multi_res["suggested_action"]
            probabilities = multi_res["probabilities"]
            alert = (rotten_count > 0 or damaged_count > 0)
            evidence_url = multi_res["annotated_image_url"]

        elif len(boxes) == 1 and not (boxes[0][2] - boxes[0][0] > w * 0.95 and boxes[0][3] - boxes[0][1] > h * 0.95):
            box = boxes[0]
            crop = img_bgr[box[1]:box[3], box[0]:box[2]]
            pred = detector.predict_single_crop(crop if crop.size > 0 else img_bgr)

            is_non_fruit = pred["label"] in ["Not Fruits", "Not Fruit"]
            total_fruits = 0 if is_non_fruit else 1
            good_count = 1 if pred["label"] == "Good" else 0
            damaged_count = 1 if pred["label"] == "Damaged" else 0
            rotten_count = 1 if pred["label"] == "Rotten" else 0
            acceptance_rate = 100.0 if pred["label"] == "Good" else 0.0
            predicted_class = pred["label"]
            confidence = pred["confidence"]
            status = pred["status"]
            suggested_action = pred["suggested_action"]
            probabilities = pred["probabilities"]
            alert = pred["label"] in ["Damaged", "Rotten"]

            detections = [{
                "x1": int(box[0]),
                "y1": int(box[1]),
                "x2": int(box[2]),
                "y2": int(box[3]),
                "label": pred["label"],
                "confidence": pred["confidence"],
                "status": pred["status"]
            }]
            annotated = draw_single_annotation(img_bgr, pred["label"], pred["confidence"], box=box)
            _, evidence_url = save_evidence_snapshot(annotated, prefix="live_stream")

        else:
            pred = detector.predict_single_crop(img_bgr)
            is_non_fruit = pred["label"] in ["Not Fruits", "Not Fruit"]
            total_fruits = 0 if is_non_fruit else 1
            good_count = 1 if pred["label"] == "Good" else 0
            damaged_count = 1 if pred["label"] == "Damaged" else 0
            rotten_count = 1 if pred["label"] == "Rotten" else 0
            acceptance_rate = 100.0 if pred["label"] == "Good" else 0.0
            predicted_class = pred["label"]
            confidence = pred["confidence"]
            status = pred["status"]
            suggested_action = pred["suggested_action"]
            probabilities = pred["probabilities"]
            alert = pred["label"] in ["Damaged", "Rotten"]

            if total_fruits > 0:
                detections = [{
                    "x1": 10,
                    "y1": 50,
                    "x2": w - 10,
                    "y2": h - 10,
                    "label": pred["label"],
                    "confidence": pred["confidence"],
                    "status": pred["status"]
                }]
                annotated = draw_single_annotation(img_bgr, pred["label"], pred["confidence"], box=(10, 50, w - 10, h - 10))
                _, evidence_url = save_evidence_snapshot(annotated, prefix="live_stream")
            else:
                detections = []
                evidence_url = None

        # Log defect alert to database if confidence >= 0.75
        if alert and confidence >= 0.75:
            inspection = FruitInspection(
                image_path=evidence_url,
                predicted_class=predicted_class,
                confidence=confidence,
                inspection_status=status,
                fruit_count=max(1, total_fruits),
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
