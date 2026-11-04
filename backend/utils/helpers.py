import os
import cv2
import uuid
import base64
import logging
import binascii
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from fastapi import UploadFile, HTTPException, status

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
EVIDENCE_DIR = os.path.join(STATIC_DIR, "evidence")
UPLOADS_DIR = os.path.join(STATIC_DIR, "uploads")
SAMPLES_DIR = os.path.join(STATIC_DIR, "samples")
PROCESSED_DIR = os.path.join(STATIC_DIR, "processed")

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB limit for uploads

def ensure_directories():
    """Ensure all static assets directories exist."""
    for path in [STATIC_DIR, EVIDENCE_DIR, UPLOADS_DIR, SAMPLES_DIR, PROCESSED_DIR]:
        os.makedirs(path, exist_ok=True)

def numpy_to_base64(img_bgr: np.ndarray) -> str:
    """Encode OpenCV BGR image into base64 JPEG."""
    _, buffer = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_str}"

def base64_to_numpy(b64_data: str) -> Optional[np.ndarray]:
    """Decode base64 image into OpenCV BGR numpy array."""
    if not b64_data or not isinstance(b64_data, str):
        return None
    try:
        if ',' in b64_data:
            b64_data = b64_data.split(',', 1)[1]
        img_bytes = base64.b64decode(b64_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except (binascii.Error, ValueError) as err:
        logger.debug(f"Base64 image decode failed: {err}")
        return None
    except Exception as err:
        logger.debug(f"Unexpected error in base64 decode: {err}")
        return None

def sanitize_csv_cell(value: Any) -> str:
    """
    Sanitize cell values against CSV / Formula Injection (DDE) per RFC 4180 & OWASP guidelines.
    Neutralizes leading formula triggers (=, +, -, @, \\t, \\r) by prepending a single quote (').
    """
    if value is None:
        return '""'
    s = str(value)
    triggers = ('=', '+', '-', '@', '\t', '\r')
    trimmed = s.lstrip()
    if trimmed and trimmed.startswith(triggers):
        s = "'" + s
    escaped = s.replace('"', '""')
    return f'"{escaped}"'

def process_and_save_upload(file: UploadFile, prefix: str = "upload") -> Tuple[str, str, np.ndarray]:
    """
    Validates uploaded file:
    1. Checks extension against ALLOWED_IMAGE_EXTENSIONS.
    2. Enforces MAX_FILE_SIZE limit (10MB).
    3. Verifies image decodability in memory via OpenCV before saving to disk.
    4. Generates a collision-free UUID filename.
    5. Confirms destination path is strictly within UPLOADS_DIR (Path traversal prevention).
    Returns: (disk_path, static_url, img_bgr)
    """
    ensure_directories()

    # 1. Extension validation
    raw_filename = file.filename or "unknown.jpg"
    raw_ext = Path(raw_filename).suffix.lower()
    if not raw_ext or raw_ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{raw_ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}"
        )

    # 2. Size limitation check (read max limit + 1 byte)
    contents = file.file.read(MAX_FILE_SIZE + 1)
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE // (1024 * 1024)}MB."
        )

    # 3. Decode in memory BEFORE saving to disk
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None or img_bgr.size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file format or corrupted image stream."
        )

    # 4. Generate random safe filename & verify sandbox directory containment
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{prefix}_{ts}_{uuid.uuid4().hex[:12]}{raw_ext}"
    target_path = os.path.abspath(os.path.join(UPLOADS_DIR, safe_filename))
    uploads_dir_abs = os.path.abspath(UPLOADS_DIR)

    # 5. Path confinement verification
    if os.path.commonpath([target_path, uploads_dir_abs]) != uploads_dir_abs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path traversal violation detected."
        )

    with open(target_path, "wb") as f:
        f.write(contents)

    upload_url = f"/static/uploads/{safe_filename}"
    return target_path, upload_url, img_bgr

def draw_single_annotation(img: np.ndarray, label: str, conf: float) -> np.ndarray:
    """Draw industrial HUD quality banner overlay for single fruit."""
    annotated = img.copy()
    h, w = annotated.shape[:2]

    # Color palette
    if label == "Good":
        color = (50, 205, 50) # Green
        status_text = "PASSED - ACCEPTED FOR PACKAGING"
    elif label == "Damaged":
        color = (30, 165, 255) # Orange / Amber
        status_text = "DEFECT - SECONDARY INSPECTION"
    elif label in ["Not Fruits", "Not Fruit"]:
        color = (180, 180, 180) # Monochrome Gray
        status_text = "REJECTED - NOT A FRUIT (NON-PRODUCE)"
    else: # Rotten
        color = (60, 60, 255) # Red
        status_text = "CRITICAL - REJECTED (SPOILAGE)"

    # Industrial top HUD banner
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

    # HUD Text
    header = f"[{label.upper()}]  CONF: {conf*100:.1f}%  |  {status_text}"
    cv2.putText(annotated, header, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Frame border indicator
    cv2.rectangle(annotated, (2, 2), (w-3, h-3), color, 3)

    return annotated

def draw_multi_annotations(img: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """Draw bounding boxes and class tags for multi-fruit inspection."""
    annotated = img.copy()
    h, w = annotated.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        label = det["label"]
        conf = det["confidence"]

        if label == "Good":
            box_color = (60, 220, 60)
            tag_bg = (30, 150, 30)
        elif label == "Damaged":
            box_color = (30, 175, 255)
            tag_bg = (20, 120, 200)
        elif label in ["Not Fruits", "Not Fruit"]:
            box_color = (180, 180, 180)
            tag_bg = (60, 60, 60)
        else: # Rotten
            box_color = (50, 50, 240)
            tag_bg = (30, 30, 180)

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

        # Label tag
        tag_str = f"[{label.upper()} {int(conf*100)}%]"
        (text_w, text_h), baseline = cv2.getTextSize(tag_str, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)

        tag_y1 = max(0, y1 - text_h - 8)
        tag_y2 = y1
        cv2.rectangle(annotated, (x1, tag_y1), (x1 + text_w + 10, tag_y2), tag_bg, -1)
        cv2.putText(annotated, tag_str, (x1 + 5, tag_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    # Top summary bar
    good_n = sum(1 for d in detections if d["label"] == "Good")
    damaged_n = sum(1 for d in detections if d["label"] == "Damaged")
    rotten_n = sum(1 for d in detections if d["label"] == "Rotten")

    summary_bar = annotated.copy()
    cv2.rectangle(summary_bar, (0, 0), (w, 36), (15, 15, 15), -1)
    cv2.addWeighted(summary_bar, 0.85, annotated, 0.15, 0, annotated)

    summary_text = f"TOTAL: {len(detections)}  |  GOOD: {good_n}  |  DAMAGED: {damaged_n}  |  ROTTEN: {rotten_n}"
    cv2.putText(annotated, summary_text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

    return annotated

def save_evidence_snapshot(img_bgr: np.ndarray, prefix: str = "inspect") -> Tuple[str, str]:
    """Save an evidence image and return (disk_path, relative_url_path)."""
    ensure_directories()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{prefix}_{ts}.jpg"
    filepath = os.path.join(EVIDENCE_DIR, filename)
    success = cv2.imwrite(filepath, img_bgr)
    if not success:
        raise IOError(f"Failed to persist evidence snapshot image to: {filepath}")
    rel_url = f"/static/evidence/{filename}"
    return filepath, rel_url
