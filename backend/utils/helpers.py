import os
import cv2
import base64
import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional

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

def _draw_hud_box(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, label: str, conf: float):
    """Draw high-contrast industrial HUD bounding box with corner brackets and label tag."""
    h, w = img.shape[:2]
    # Bound coordinates
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))

    if label == "Good":
        box_color = (60, 220, 60)      # High-vis green
        tag_bg = (20, 140, 20)
    elif label == "Damaged":
        box_color = (30, 180, 255)     # Amber / orange
        tag_bg = (20, 120, 200)
    elif label in ["Not Fruits", "Not Fruit"]:
        box_color = (180, 180, 180)   # Neutral monochrome gray
        tag_bg = (60, 60, 60)
    else: # Rotten
        box_color = (50, 50, 240)      # High-vis red
        tag_bg = (20, 20, 180)

    # 1. Main bounding rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)

    # 2. Corner brackets for precision industrial look
    corner_len = max(6, min(24, (x2 - x1) // 5, (y2 - y1) // 5))
    corner_color = (255, 255, 255)
    # Top-left
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), corner_color, 3)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), corner_color, 3)
    # Top-right
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), corner_color, 3)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), corner_color, 3)
    # Bottom-left
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), corner_color, 3)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), corner_color, 3)
    # Bottom-right
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), corner_color, 3)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), corner_color, 3)

    # 3. Label tag
    tag_str = f"[{label.upper()} {int(conf * 100)}%]"
    (tw, th), _ = cv2.getTextSize(tag_str, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)

    if y1 > th + 10:
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 10, y1), tag_bg, -1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 10, y1), box_color, 1)
        cv2.putText(img, tag_str, (x1 + 5, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    else:
        cv2.rectangle(img, (x1, y1), (x1 + tw + 10, y1 + th + 8), tag_bg, -1)
        cv2.rectangle(img, (x1, y1), (x1 + tw + 10, y1 + th + 8), box_color, 1)
        cv2.putText(img, tag_str, (x1 + 5, y1 + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

def draw_single_annotation(
    img: np.ndarray, 
    label: str, 
    conf: float,
    box: Optional[Tuple[int, int, int, int]] = None
) -> np.ndarray:
    """Draw industrial HUD quality banner overlay and visible bounding box for single fruit."""
    annotated = img.copy()
    h, w = annotated.shape[:2]

    # Draw visible fruit bounding box
    if box is not None:
        _draw_hud_box(annotated, box[0], box[1], box[2], box[3], label, conf)
    else:
        # Default frame box
        _draw_hud_box(annotated, 8, 55, w - 8, h - 8, label, conf)

    # Industrial top HUD banner
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 48), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

    if label == "Good":
        color = (50, 205, 50)
        status_text = "PASSED - ACCEPTED FOR PACKAGING"
    elif label == "Damaged":
        color = (30, 165, 255)
        status_text = "DEFECT - SECONDARY INSPECTION"
    elif label in ["Not Fruits", "Not Fruit"]:
        color = (180, 180, 180)
        status_text = "REJECTED - NOT A FRUIT (NON-PRODUCE)"
    else: # Rotten
        color = (60, 60, 255)
        status_text = "CRITICAL - REJECTED (SPOILAGE)"

    header = f"[{label.upper()}]  CONF: {conf*100:.1f}%  |  {status_text}"
    cv2.putText(annotated, header, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    cv2.rectangle(annotated, (0, 48), (w, 50), color, -1)

    return annotated

def draw_multi_annotations(img: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """Draw bounding boxes, corner brackets, and class tags for multi-fruit inspection."""
    annotated = img.copy()
    h, w = annotated.shape[:2]

    # Draw box for each fruit
    for det in detections:
        _draw_hud_box(
            annotated, 
            det["x1"], 
            det["y1"], 
            det["x2"], 
            det["y2"], 
            det["label"], 
            det["confidence"]
        )

    # Top summary bar
    good_n = sum(1 for d in detections if d["label"] == "Good")
    damaged_n = sum(1 for d in detections if d["label"] == "Damaged")
    rotten_n = sum(1 for d in detections if d["label"] == "Rotten")
    not_fruit_n = sum(1 for d in detections if d["label"] in ["Not Fruits", "Not Fruit"])

    summary_bar = annotated.copy()
    cv2.rectangle(summary_bar, (0, 0), (w, 36), (15, 15, 15), -1)
    cv2.addWeighted(summary_bar, 0.88, annotated, 0.12, 0, annotated)

    summary_text = f"TOTAL: {len(detections)}  |  GOOD: {good_n}  |  DAMAGED: {damaged_n}  |  ROTTEN: {rotten_n}"
    if not_fruit_n > 0:
        summary_text += f"  |  NON-PRODUCE: {not_fruit_n}"
    cv2.putText(annotated, summary_text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

    # Subtly accent bottom line of summary bar
    bar_color = (60, 220, 60) if rotten_n == 0 and damaged_n == 0 else (50, 50, 240) if rotten_n > 0 else (30, 180, 255)
    cv2.line(annotated, (0, 36), (w, 36), bar_color, 2)

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
