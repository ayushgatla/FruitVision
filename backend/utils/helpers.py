import os
import cv2
import base64
import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
EVIDENCE_DIR = os.path.join(STATIC_DIR, "evidence")
UPLOADS_DIR = os.path.join(STATIC_DIR, "uploads")
SAMPLES_DIR = os.path.join(STATIC_DIR, "samples")
PROCESSED_DIR = os.path.join(STATIC_DIR, "processed")

def ensure_directories():
    """Ensure all static assets directories exist."""
    for path in [STATIC_DIR, EVIDENCE_DIR, UPLOADS_DIR, SAMPLES_DIR, PROCESSED_DIR]:
        os.makedirs(path, exist_ok=True)

def numpy_to_base64(img_bgr: np.ndarray) -> str:
    """Encode OpenCV BGR image into base64 JPEG."""
    _, buffer = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_str}"

def base64_to_numpy(b64_data: str) -> np.ndarray:
    """Decode base64 image into OpenCV BGR numpy array."""
    if ',' in b64_data:
        b64_data = b64_data.split(',', 1)[1]
    img_bytes = base64.b64decode(b64_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

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
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{prefix}_{ts}.jpg"
    filepath = os.path.join(EVIDENCE_DIR, filename)
    cv2.imwrite(filepath, img_bgr)
    rel_url = f"/static/evidence/{filename}"
    return filepath, rel_url

