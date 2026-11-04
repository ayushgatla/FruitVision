import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import threading
import logging
from PIL import Image
from torchvision import models, transforms
from typing import Dict, Any, List, Tuple, Optional

from backend.utils.helpers import (
    draw_single_annotation,
    draw_multi_annotations,
    save_evidence_snapshot,
    STATIC_DIR
)
from training.preprocessing import inference_transform

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
CUSTOM_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "best.pt")
ALT_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "fruit_efficientnet_b0.pt")

CANONICAL_CLASSES_3 = ["Good", "Damaged", "Rotten"]
DEFAULT_CLASSES_4 = ["Apple_Bad", "Apple_Good", "Banana_Bad", "Banana_Good"]

PRODUCE_SET = {
    'banana', 'orange', 'lemon', 'fig', 'pineapple', 'jackfruit', 'custard apple',
    'pomegranate', 'strawberry', 'granny smith', 'bell pepper', 'cucumber', 'acorn squash',
    'butternut squash', 'spaghetti squash', 'head cabbage', 'broccoli', 'cauliflower',
    'zucchini', 'mushroom', 'cardoon', 'artichoke', 'hip', 'rose hip', 'acorn'
}

NON_PRODUCE_TRIGGERS = [
    'car', 'truck', 'bus', 'vehicle', 'bicycle', 'motorcycle', 'airplane', 'boat',
    'dog', 'cat', 'horse', 'bird', 'cellular telephone', 'phone', 'laptop', 'computer',
    'keyboard', 'mouse', 'screen', 'television', 'monitor', 'shoe', 'sneaker', 'sandal',
    'boot', 'suit', 'jersey', 'desk', 'chair', 'table', 'couch', 'bottle', 'cup', 'mug',
    'pen', 'pencil', 'scissors', 'screwdriver', 'web site', 'menu', 'envelope', 'packet',
    'book', 'comic book', 'binder', 'cellular', 'notebook', 'wallet', 'clock', 'watch'
]

class FruitVisionDetector:
    _instance = None

    def __init__(self, weights_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_lock = threading.RLock()
        self.classes = CANONICAL_CLASSES_3
        self.model_version = "EfficientNetB0-TransferLearning-v2.0"

        # Validate weights path against path traversal
        if weights_path:
            resolved_target = os.path.abspath(weights_path)
            weights_dir_abs = os.path.abspath(WEIGHTS_DIR)
            if os.path.commonpath([resolved_target, weights_dir_abs]) != weights_dir_abs or not resolved_target.endswith((".pt", ".pth")):
                raise ValueError(f"Path traversal violation or invalid weights file: {weights_path}")

        self.weights_path = weights_path or (
            CUSTOM_WEIGHTS_PATH if os.path.exists(CUSTOM_WEIGHTS_PATH)
            else (ALT_WEIGHTS_PATH if os.path.exists(ALT_WEIGHTS_PATH) else None)
        )
        self.using_custom_weights = False
        self.model = None

        # Pretrained ImageNet reference model for produce vs non-produce verification
        try:
            self.imagenet_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT).eval().to(self.device)
            self.imagenet_cats = models.EfficientNet_B0_Weights.DEFAULT.meta["categories"]
        except Exception as e:
            logger.info("Notice: Pretrained ImageNet reference model init: %s", e)
            self.imagenet_model = None
            self.imagenet_cats = []

        self._load_model()

    def _load_model(self):
        """Build and load EfficientNetB0 architecture dynamically matching checkpoint classes."""
        with self.model_lock:
            logger.info("Loading EfficientNetB0 on %s...", self.device)
            try:
                target_weights = self.weights_path or (
                    CUSTOM_WEIGHTS_PATH if os.path.exists(CUSTOM_WEIGHTS_PATH)
                    else (ALT_WEIGHTS_PATH if os.path.exists(ALT_WEIGHTS_PATH) else None)
                )

                ckpt = None
                classes = CANONICAL_CLASSES_3
                if target_weights and os.path.exists(target_weights):
                    logger.info("Reading checkpoint metadata from %s...", target_weights)
                    ckpt = torch.load(target_weights, map_location=self.device, weights_only=True)
                    if isinstance(ckpt, dict) and "classes" in ckpt:
                        classes = ckpt["classes"]
                    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                        state = ckpt["model_state_dict"]
                        for k in ["classifier.4.weight", "classifier.1.weight"]:
                            if k in state:
                                out_size = state[k].shape[0]
                                if out_size == 4:
                                    classes = DEFAULT_CLASSES_4
                                elif out_size == 3:
                                    classes = CANONICAL_CLASSES_3
                                break
                    if isinstance(ckpt, dict) and "model_version" in ckpt:
                        self.model_version = ckpt["model_version"]

                self.classes = classes
                logger.info("Active model classes (%d): %s", len(self.classes), self.classes)

                # Build base model and classification head
                base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
                in_features = base_model.classifier[1].in_features
                base_model.classifier = nn.Sequential(
                    nn.Dropout(p=0.3, inplace=True),
                    nn.Linear(in_features, 128),
                    nn.ReLU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(128, len(self.classes))
                )

                if ckpt is not None:
                    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                        base_model.load_state_dict(ckpt["model_state_dict"])
                    else:
                        base_model.load_state_dict(ckpt)
                    self.using_custom_weights = True
                    self.weights_path = target_weights
                    logger.info("Custom trained weights loaded successfully.")
                else:
                    self.using_custom_weights = False
                    logger.info("Pretrained ImageNet EfficientNetB0 backbone ready.")

                base_model.to(self.device)
                base_model.eval()
                self.model = base_model

            except Exception as e:
                logger.error("Error loading EfficientNetB0 model: %s", e)
                self.model = None

    def reload_custom_weights(self) -> bool:
        """Hot-reload weights without restarting the FastAPI server."""
        with self.model_lock:
            if os.path.exists(CUSTOM_WEIGHTS_PATH):
                self.weights_path = CUSTOM_WEIGHTS_PATH
                self._load_model()
                return True
            elif os.path.exists(ALT_WEIGHTS_PATH):
                self.weights_path = ALT_WEIGHTS_PATH
                self._load_model()
                return True
            return False

    def _analyze_surface_defect_heuristics(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Optical surface defect & rot analysis.
        Inspects necrotic lesions, mold coverage, and skin degradation.
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # Mask fruit from light background
        fruit_mask = (gray < 240) & (gray > 15)
        fruit_pixels = int(np.sum(fruit_mask))
        if fruit_pixels < 400:
            return {"override": False, "label": "Good", "confidence": 0.9}

        # 1. Vibrant fresh peel detection
        yellow_peel = (hsv[:,:,0] >= 17) & (hsv[:,:,0] <= 36) & (hsv[:,:,1] >= 80) & (hsv[:,:,2] >= 110) & fruit_mask
        orange_peel = (hsv[:,:,0] >= 9) & (hsv[:,:,0] <= 18) & (hsv[:,:,1] >= 130) & (hsv[:,:,2] >= 140) & fruit_mask
        red_peel = ((hsv[:,:,0] <= 8) | (hsv[:,:,0] >= 165)) & (hsv[:,:,1] >= 100) & (hsv[:,:,2] >= 120) & fruit_mask
        green_peel = (hsv[:,:,0] >= 35) & (hsv[:,:,0] <= 80) & (hsv[:,:,1] >= 70) & (hsv[:,:,2] >= 90) & fruit_mask

        vibrant_peel = yellow_peel | orange_peel | red_peel | green_peel
        vibrant_ratio = float(np.sum(vibrant_peel)) / fruit_pixels

        # 2. Necrotic rot & decay detection
        necrotic_brown = (hsv[:,:,0] >= 2) & (hsv[:,:,0] <= 24) & (hsv[:,:,1] >= 45) & (hsv[:,:,2] >= 25) & (hsv[:,:,2] <= 175) & (~yellow_peel) & (~orange_peel) & fruit_mask
        dark_necrosis = (hsv[:,:,2] < 45) & fruit_mask
        rot_mask = necrotic_brown | dark_necrosis
        rot_ratio = float(np.sum(rot_mask)) / fruit_pixels

        # Severe necrotic rot
        if rot_ratio >= 0.25 or (rot_ratio >= 0.15 and vibrant_ratio < 0.50):
            conf = float(round(max(0.94, min(0.995, 0.85 + rot_ratio * 0.25)), 4))
            return {"override": True, "label": "Rotten", "confidence": conf, "rot_ratio": rot_ratio, "vibrant_ratio": vibrant_ratio}

        # Localized surface damage / bruise
        if rot_ratio >= 0.08 and rot_ratio < 0.25:
            conf = float(round(max(0.88, min(0.96, 0.80 + rot_ratio * 0.7)), 4))
            return {"override": True, "label": "Damaged", "confidence": conf, "rot_ratio": rot_ratio, "vibrant_ratio": vibrant_ratio}

        # Fresh peel with low rot
        if vibrant_ratio >= 0.60 and rot_ratio < 0.08:
            conf = float(round(max(0.95, min(0.99, 0.90 + vibrant_ratio * 0.1)), 4))
            return {"override": True, "label": "Good", "confidence": conf, "rot_ratio": rot_ratio, "vibrant_ratio": vibrant_ratio}

        return {"override": False, "rot_ratio": rot_ratio, "vibrant_ratio": vibrant_ratio}

    def verify_is_fruit(self, img_bgr: np.ndarray) -> Tuple[bool, str, float]:
        """Verify if the input image depicts a genuine organic fruit / produce item."""
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        total_pixels = h * w

        non_bg = (gray < 242) & (gray > 12)
        total_non_bg = int(np.sum(non_bg))
        if total_non_bg < 350:
            return False, "Blank image or background only", 0.99

        edges = cv2.Canny(gray, 100, 200)
        edge_ratio = float(np.sum(edges > 0)) / total_pixels
        if edge_ratio > 0.16 and total_non_bg > 0.75 * total_pixels:
            return False, "Document or text screenshot detected", 0.98

        organic_hues = (
            ((hsv[:,:,0] >= 0) & (hsv[:,:,0] <= 85)) | 
            ((hsv[:,:,0] >= 160) & (hsv[:,:,0] <= 179))
        )
        organic_pigment = organic_hues & (hsv[:,:,1] >= 20) & non_bg

        stark_blue_cyan = (hsv[:,:,0] >= 90) & (hsv[:,:,0] <= 135) & (hsv[:,:,1] >= 35) & non_bg
        blue_ratio = float(np.sum(stark_blue_cyan)) / max(total_non_bg, 1)
        organic_ratio = float(np.sum(organic_pigment)) / max(total_non_bg, 1)

        if blue_ratio > 0.20:
            return False, "Inorganic chromatic signature (Blue / Cyan tones)", 0.98

        if organic_ratio < 0.08:
            return False, "Missing biological produce pigments", 0.97

        # Zero-shot ImageNet Check
        if self.imagenet_model is not None and self.imagenet_cats:
            try:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                pil_im = Image.fromarray(img_rgb)
                t = inference_transform(pil_im).unsqueeze(0).to(self.device)
                with self.model_lock:
                    with torch.no_grad():
                        out = torch.softmax(self.imagenet_model(t), dim=1)[0]
                top_prob, top_idx = torch.max(out, dim=0)
                top_cat = self.imagenet_cats[top_idx.item()].lower()

                top5_indices = torch.topk(out, 5).indices.tolist()
                has_produce_in_top5 = any(self.imagenet_cats[idx].lower() in PRODUCE_SET for idx in top5_indices)
                is_explicit_non_fruit = any(trigger in top_cat for trigger in NON_PRODUCE_TRIGGERS)

                if is_explicit_non_fruit and top_prob.item() > 0.30 and not has_produce_in_top5:
                    return False, f"Identified non-fruit object ({top_cat})", float(round(top_prob.item(), 4))
            except Exception as e:
                logger.debug("ImageNet zero-shot produce check exception: %s", e)

        return True, "Organic fruit verified", 0.98

    def predict_single_crop(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """Perform EfficientNetB0 inference on an individual fruit crop with dual-tier canonical mapping."""
        # Out-Of-Distribution Non-Fruit check
        is_fruit_ok, reject_reason, fruit_conf = self.verify_is_fruit(img_bgr)
        if not is_fruit_ok:
            conf_val = float(round(max(0.96, fruit_conf), 4))
            return {
                "label": "Not Fruits",
                "raw_class": "Not Fruits",
                "fruit_type": "Unknown",
                "confidence": conf_val,
                "probabilities": {
                    "Not Fruits": conf_val,
                    "Good": 0.0,
                    "Damaged": 0.0,
                    "Rotten": float(round(1.0 - conf_val, 4))
                },
                "status": "NON_PRODUCE",
                "suggested_action": f"Rejected: Not a Fruit ({reject_reason})"
            }

        if self.model is None:
            self._load_model()

        # Preprocessing
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tensor = inference_transform(pil_img).unsqueeze(0).to(self.device)

        with self.model_lock:
            with torch.no_grad():
                outputs = self.model(tensor)
                probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
            classes_snapshot = list(self.classes)

        prob_dict = {
            classes_snapshot[i]: float(round(probs[i], 4))
            for i in range(len(classes_snapshot))
        }

        pred_idx = int(np.argmax(probs))
        raw_class = classes_snapshot[pred_idx]
        confidence = float(round(probs[pred_idx], 4))

        # Dual-tier canonical quality and fruit mapping
        if any("_" in c for c in classes_snapshot):
            # 4-class model: Apple_Bad, Apple_Good, Banana_Bad, Banana_Good
            fruit_type = "Apple" if "apple" in raw_class.lower() else ("Banana" if "banana" in raw_class.lower() else "Produce")
            is_good = "good" in raw_class.lower()

            if is_good:
                label = "Good"
                status = "ACCEPTED"
                action = "Direct to Automated Packaging Line (Market Grade A)"
            else:
                label = "Rotten"
                status = "REJECTED"
                action = "Immediate Diverter Ejection to Bio-Waste / Composting"
        else:
            # 3-class model: Good, Damaged, Rotten
            label = raw_class
            fruit_type = "Produce"
            if label == "Good":
                status = "ACCEPTED"
                action = "Direct to Automated Packaging Line (Market Grade A)"
            elif label == "Damaged":
                status = "SECONDARY_INSPECTION"
                action = "Route to Secondary Manual Grading / Reprocessing Unit"
            else:
                status = "REJECTED"
                action = "Immediate Diverter Ejection to Bio-Waste / Composting"

        # Food safety heuristic:
        # Optical analysis detects surface lesions, necrosis, and bruising.
        # Under ISO 22000 / HACCP: If physical damage or rot is detected, never allow it to pass as Good!
        guard = self._analyze_surface_defect_heuristics(img_bgr)
        if guard.get("override"):
            if guard["label"] in ["Rotten", "Damaged"]:
                label = guard["label"]
                confidence = guard["confidence"]
            elif not self.using_custom_weights:
                label = guard["label"]
                confidence = guard["confidence"]

        # Update status and action according to final label
        if label == "Good":
            status = "ACCEPTED"
            action = "Direct to Automated Packaging Line (Market Grade A)"
        elif label == "Damaged":
            status = "SECONDARY_INSPECTION"
            action = "Route to Secondary Manual Grading / Reprocessing Unit"
        elif label == "Rotten":
            status = "REJECTED"
            action = "Immediate Diverter Ejection to Bio-Waste / Composting"

        return {
            "label": label,
            "raw_class": raw_class,
            "fruit_type": fruit_type,
            "confidence": confidence,
            "probabilities": prob_dict,
            "status": status,
            "suggested_action": action
        }

    def detect_and_classify_image(self, img_bgr: np.ndarray, source_id: str = "upload") -> Dict[str, Any]:
        """Classify a single fruit image and generate annotated HUD evidence."""
        pred = self.predict_single_crop(img_bgr)
        display_label = pred["raw_class"] if "_" in pred.get("raw_class", "") else pred["label"]
        annotated = draw_single_annotation(img_bgr, display_label, pred["confidence"])
        
        _, evidence_url = save_evidence_snapshot(annotated, prefix=f"single_{pred['label'].lower()}")

        return {
            "predicted_class": pred["label"],
            "raw_class": pred["raw_class"],
            "fruit_type": pred.get("fruit_type", "Produce"),
            "confidence": pred["confidence"],
            "status": pred["status"],
            "suggested_action": pred["suggested_action"],
            "probabilities": pred["probabilities"],
            "annotated_image_url": evidence_url,
            "device": str(self.device).upper(),
            "model_version": self.model_version
        }

    def detect_multiple_fruits(self, img_bgr: np.ndarray, min_area: int = 1200) -> Dict[str, Any]:
        """Locate multiple fruits using computer vision segmentation and run classification."""
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw > w * 0.95 and bh > h * 0.95:
                continue

            pad_x = int(bw * 0.08)
            pad_y = int(bh * 0.08)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + bw + pad_x)
            y2 = min(h, y + bh + pad_y)

            crop = img_bgr[y1:y2, x1:x2]
            if crop.shape[0] < 20 or crop.shape[1] < 20:
                continue

            pred = self.predict_single_crop(crop)
            detections.append({
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "label": pred["label"],
                "raw_class": pred["raw_class"],
                "confidence": pred["confidence"],
                "status": pred["status"]
            })

        # Fallback if no distinct multiple contours found
        if len(detections) == 0:
            pred = self.predict_single_crop(img_bgr)
            detections.append({
                "x1": 10,
                "y1": 10,
                "x2": w - 10,
                "y2": h - 10,
                "label": pred["label"],
                "raw_class": pred["raw_class"],
                "confidence": pred["confidence"],
                "status": pred["status"]
            })

        good_count = sum(1 for d in detections if d["label"] == "Good")
        damaged_count = sum(1 for d in detections if d["label"] == "Damaged")
        rotten_count = sum(1 for d in detections if d["label"] == "Rotten")
        non_fruit_count = sum(1 for d in detections if d.get("label") == "Not Fruits")
        total_fruits = len(detections)
        acceptance_rate = round((good_count / total_fruits) * 100, 1) if total_fruits > 0 else 0.0

        if rotten_count > 0:
            overall_status = "CRITICAL_DEFECTS_DETECTED"
        elif damaged_count > 0:
            overall_status = "SECONDARY_GRADING_REQUIRED"
        elif non_fruit_count > 0 and good_count == 0:
            overall_status = "INSPECTION_FAILED"
        else:
            overall_status = "BATCH_ACCEPTED"

        annotated = draw_multi_annotations(img_bgr, detections)
        _, evidence_url = save_evidence_snapshot(annotated, prefix="multi_batch")

        return {
            "total_fruits": total_fruits,
            "good_count": good_count,
            "damaged_count": damaged_count,
            "rotten_count": rotten_count,
            "acceptance_rate": acceptance_rate,
            "overall_status": overall_status,
            "detections": detections,
            "annotated_image_url": evidence_url
        }

_detector_instance: Optional[FruitVisionDetector] = None
_detector_init_lock = threading.Lock()

def get_detector() -> FruitVisionDetector:
    global _detector_instance
    if _detector_instance is None:
        with _detector_init_lock:
            if _detector_instance is None:
                _detector_instance = FruitVisionDetector()
    return _detector_instance
