import os
import logging
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any
from pydantic import BaseModel

from backend.database.database import get_db
from backend.database.models import FruitInspection
from backend.services.detector import get_detector
from backend.utils.helpers import (
    base64_to_numpy,
    ensure_directories,
    process_and_save_upload,
    draw_single_annotation,
    save_evidence_snapshot,
)
from backend.models.schemas import SingleInspectionResponse, MultiFruitInspectionResponse

logger = logging.getLogger(__name__)

router = APIRouter()


class FramePayload(BaseModel):
    frame_data: str
    source_id: Optional[str] = "webcam"


@router.post("/image", response_model=SingleInspectionResponse)
@router.post("/detect/image", response_model=SingleInspectionResponse)
def inspect_single_image(
    file: UploadFile = File(...),
    notes: Optional[str] = Form(None),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Inspect and classify a single fruit image using EfficientNetB0.
    Runs synchronously in Starlette threadpool worker to avoid blocking asyncio event loop.
    """
    ensure_directories()
    detector = get_detector()

    # Process and validate uploaded file (sanitizes filename, validates extension & size, decodes in memory)
    upload_path, upload_url, img_bgr = process_and_save_upload(file, prefix="upload")

    # Perform detection and classification
    safe_source_name = os.path.basename(file.filename or "upload.jpg")
    result = detector.detect_and_classify_image(img_bgr, source_id=safe_source_name)

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
        notes=notes or f"Inspection of {safe_source_name}",
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
def inspect_multiple_fruits(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Detect and classify multiple fruits in a single image.
    Runs synchronously in Starlette threadpool worker to avoid blocking asyncio event loop.
    """
    ensure_directories()
    detector = get_detector()

    # Process and validate uploaded file
    upload_path, upload_url, img_bgr = process_and_save_upload(file, prefix="multi")

    # Perform multi-fruit inspection
    multi_res = detector.detect_multiple_fruits(img_bgr)

    # Determine primary status
    primary_pred = "Good"
    if multi_res["rotten_count"] > 0:
        primary_pred = "Rotten"
    elif multi_res["damaged_count"] > 0:
        primary_pred = "Damaged"

    # Save to SQLite database
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
def inspect_live_frame(payload: FramePayload, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Process live video stream or webcam frame with EfficientNetB0 HUD.
    Runs synchronously in Starlette threadpool worker to avoid blocking asyncio event loop.
    """
    img_bgr = base64_to_numpy(payload.frame_data)
    if img_bgr is None or img_bgr.size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid frame: base64 payload could not be decoded as an image."
        )

    try:
        detector = get_detector()
        pred = detector.predict_single_crop(img_bgr)

        # Log defect alert to database if confidence is high (> 80%)
        if pred["label"] in ["Damaged", "Rotten"] and pred["confidence"] >= 0.80:
            evidence_url = None
            try:
                annotated = draw_single_annotation(img_bgr, pred["label"], pred["confidence"])
                _, evidence_url = save_evidence_snapshot(annotated, prefix="live_alert")
            except Exception as e:
                logger.warning(f"Could not persist live evidence snapshot: {e}")

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
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error processing live video frame: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the video frame."
        )
