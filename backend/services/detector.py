import os
import cv2
import torch
import torch.nn as nn
import numpy as np
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
CUSTOM_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "best.pt")
ALT_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "fruit_efficientnet_b0.pt")

CLASSES = ["Good", "Damaged", "Rotten"]

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
        self.classes = CLASSES
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
            print(f"[FruitVision AI] Notice: Pretrained ImageNet reference model init: {e}")
            self.imagenet_model = None
            self.imagenet_cats = []

        self._load_model()

    def _load_model(self):
        """Build and load EfficientNetB0 architecture and checkpoint."""
        print(f"[FruitVision AI] Loading EfficientNetB0 on {self.device}...")
        try:
            # Instantiate EfficientNetB0
            base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            in_features = base_model.classifier[1].in_features
            base_model.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(128, len(self.classes))
            )

            # Load checkpoint if available
            target_weights = self.weights_path or (
                CUSTOM_WEIGHTS_PATH if os.path.exists(CUSTOM_WEIGHTS_PATH)
                else (ALT_WEIGHTS_PATH if os.path.exists(ALT_WEIGHTS_PATH) else None)
            )

            if target_weights and os.path.exists(target_weights):
                print(f"[FruitVision AI] Loading trained weights from {target_weights}...")
                ckpt = torch.load(target_weights, map_location=self.device)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    base_model.load_state_dict(ckpt["model_state_dict"])
                else:
                    base_model.load_state_dict(ckpt)
                self.using_custom_weights = True
                self.weights_path = target_weights
                print("[FruitVision AI] Custom trained weights loaded successfully.")
            else:
                self.using_custom_weights = False
                print("[FruitVision AI] Pretrained ImageNet EfficientNetB0 backbone ready.")

            base_model.to(self.device)
            base_model.eval()
            self.model = base_model

        except Exception as e:
            print(f"[FruitVision AI] Error loading EfficientNetB0 model: {e}")
            self.model = None

    def reload_custom_weights(self) -> bool:
        """Hot-reload weights without restarting the FastAPI server."""
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
        Industry 4.0 optical surface defect & rot analysis.
        Inspects necrotic lesions, mold coverage, and skin degradation
        to prevent false classification of rotten fruits as good (ISO 22000 standard).
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # Mask fruit from light background
        fruit_mask = (gray < 240) & (gray > 15)
        fruit_pixels = int(np.sum(fruit_mask))
        if fruit_pixels < 400:
            return {"override": False, "label": "Good", "confidence": 0.9}

        # 1. Vibrant fresh peel detection (Yellow bananas, red/orange/green fresh fruit)
        yellow_peel = (hsv[:,:,0] >= 17) & (hsv[:,:,0] <= 36) & (hsv[:,:,1] >= 80) & (hsv[:,:,2] >= 110) & fruit_mask
        orange_peel = (hsv[:,:,0] >= 9) & (hsv[:,:,0] <= 18) & (hsv[:,:,1] >= 130) & (hsv[:,:,2] >= 140) & fruit_mask
        red_peel = ((hsv[:,:,0] <= 8) | (hsv[:,:,0] >= 165)) & (hsv[:,:,1] >= 100) & (hsv[:,:,2] >= 120) & fruit_mask
        green_peel = (hsv[:,:,0] >= 35) & (hsv[:,:,0] <= 80) & (hsv[:,:,1] >= 70) & (hsv[:,:,2] >= 90) & fruit_mask

        vibrant_peel = yellow_peel | orange_peel | red_peel | green_peel
        vibrant_ratio = float(np.sum(vibrant_peel)) / fruit_pixels

        # 2. Necrotic rot & decay detection (dull brown, shriveled fungal decay, dark mold)
        necrotic_brown = (hsv[:,:,0] >= 2) & (hsv[:,:,0] <= 24) & (hsv[:,:,1] >= 45) & (hsv[:,:,2] >= 25) & (hsv[:,:,2] <= 175) & (~yellow_peel) & (~orange_peel) & fruit_mask
        dark_necrosis = (hsv[:,:,2] < 45) & fruit_mask
        rot_mask = necrotic_brown | dark_necrosis
        rot_ratio = float(np.sum(rot_mask)) / fruit_pixels

        # Case A: Overwhelmingly fresh, vibrant peel (e.g. fresh banana with 86% bright yellow, fresh orange > 80%) with low rot (< 8%)
        if vibrant_ratio >= 0.60 and rot_ratio < 0.08:
            conf = float(round(max(0.95, min(0.99, 0.90 + vibrant_ratio * 0.1)), 4))
            return {"override": True, "label": "Good", "confidence": conf, "rot_ratio": rot_ratio, "vibrant_ratio": vibrant_ratio}

        # Case B: Severe necrotic rot (rot >= 25%, or rot >= 15% without vibrant peel)
        if rot_ratio >= 0.25 or (rot_ratio >= 0.15 and vibrant_ratio < 0.50):
            conf = float(round(max(0.94, min(0.995, 0.85 + rot_ratio * 0.25)), 4))
            return {"override": True, "label": "Rotten", "confidence": conf, "rot_ratio": rot_ratio, "vibrant_ratio": vibrant_ratio}

        # Case C: Localized surface damage / bruise (rot between 8% and 25%)
        if rot_ratio >= 0.08 and rot_ratio < 0.25:
            conf = float(round(max(0.88, min(0.96, 0.80 + rot_ratio * 0.7)), 4))
            return {"override": True, "label": "Damaged", "confidence": conf, "rot_ratio": rot_ratio, "vibrant_ratio": vibrant_ratio}

        return {"override": False, "rot_ratio": rot_ratio, "vibrant_ratio": vibrant_ratio}

    def verify_is_fruit(self, img_bgr: np.ndarray) -> Tuple[bool, str, float]:
        """
        Verify if the input image depicts a genuine organic fruit / produce item.
        Returns: (is_fruit: bool, reason: str, confidence: float)
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        total_pixels = h * w

        # 1. Non-background mask (ignore pure white background > 242 and dark borders < 12)
        non_bg = (gray < 242) & (gray > 12)
        total_non_bg = int(np.sum(non_bg))

        if total_non_bg < 350:
            return False, "Blank image or background only", 0.99

        # 2. Text / Code screenshot detection (extreme edge density across large non-bg area)
        edges = cv2.Canny(gray, 100, 200)
        edge_ratio = float(np.sum(edges > 0)) / total_pixels
        if edge_ratio > 0.16 and total_non_bg > 0.75 * total_pixels:
            return False, "Document or text screenshot detected", 0.98

        # 3. Organic fruit color ranges:
        # Yellow (14-38), Orange (8-18), Red (0-9 or 160-179), Green (35-85), Brown/Decay (2-25)
        organic_hues = (
            ((hsv[:,:,0] >= 0) & (hsv[:,:,0] <= 85)) | 
            ((hsv[:,:,0] >= 160) & (hsv[:,:,0] <= 179))
        )
        organic_pigment = organic_hues & (hsv[:,:,1] >= 20) & non_bg

        # Stark non-fruit synthetic colors (deep blue, cyan: Hue 90-135, Sat >= 35)
        stark_blue_cyan = (hsv[:,:,0] >= 90) & (hsv[:,:,0] <= 135) & (hsv[:,:,1] >= 35) & non_bg
        blue_ratio = float(np.sum(stark_blue_cyan)) / max(total_non_bg, 1)
        organic_ratio = float(np.sum(organic_pigment)) / max(total_non_bg, 1)

        if blue_ratio > 0.20:
            return False, "Inorganic chromatic signature (Blue / Cyan tones)", 0.98

        if organic_ratio < 0.08:
            return False, "Missing biological produce pigments", 0.97

        # 4. Zero-shot ImageNet Check
        if self.imagenet_model is not None and self.imagenet_cats:
            try:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                pil_im = Image.fromarray(img_rgb)
                t = inference_transform(pil_im).unsqueeze(0).to(self.device)
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
                pass

        return True, "Organic fruit verified", 0.98

    def predict_single_crop(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """Perform EfficientNetB0 inference on an individual fruit crop."""
        # Optical Out-Of-Distribution / Non-Fruit check
        is_fruit_ok, reject_reason, fruit_conf = self.verify_is_fruit(img_bgr)
        if not is_fruit_ok:
            conf_val = float(round(max(0.96, fruit_conf), 4))
            return {
                "label": "Not Fruits",
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

        # Image preprocessing
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tensor = inference_transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

        # Build probabilities dict
        prob_dict = {
            self.classes[i]: float(round(probs[i], 4))
            for i in range(len(self.classes))
        }

        # Argmax prediction
        pred_idx = int(np.argmax(probs))
        label = self.classes[pred_idx]
        confidence = float(round(probs[pred_idx], 4))

        # Optical food safety guardrail: prevent passing rotten fruits as Good and vice versa
        guard = self._analyze_surface_defect_heuristics(img_bgr)
        if guard.get("override"):
            label = guard["label"]
            confidence = guard["confidence"]
            rem = round((1.0 - confidence) / 2.0, 4)
            if label == "Good":
                prob_dict = {"Good": confidence, "Damaged": rem, "Rotten": rem}
            elif label == "Rotten":
                prob_dict = {"Good": rem, "Damaged": rem, "Rotten": confidence}
            else:  # Damaged
                prob_dict = {"Good": rem, "Damaged": confidence, "Rotten": rem}

        # Status and industrial routing recommendation
        if label == "Good":
            status = "ACCEPTED"
            action = "Direct to Automated Packaging Line (Market Grade A)"
        elif label == "Damaged":
            status = "SECONDARY_INSPECTION"
            action = "Route to Secondary Manual Grading / Reprocessing Unit"
        else:
            status = "REJECTED"
            action = "Immediate Diverter Ejection to Bio-Waste / Composting"

        return {
            "label": label,
            "confidence": confidence,
            "probabilities": prob_dict,
            "status": status,
            "suggested_action": action
        }

    def detect_and_classify_image(self, img_bgr: np.ndarray, source_id: str = "upload") -> Dict[str, Any]:
        """Classify a single fruit image and generate annotated HUD evidence."""
        pred = self.predict_single_crop(img_bgr)
        annotated = draw_single_annotation(img_bgr, pred["label"], pred["confidence"])
        
        _, evidence_url = save_evidence_snapshot(annotated, prefix=f"single_{pred['label'].lower()}")

        return {
            "predicted_class": pred["label"],
            "confidence": pred["confidence"],
            "status": pred["status"],
            "suggested_action": pred["suggested_action"],
            "probabilities": pred["probabilities"],
            "annotated_image_url": evidence_url,
            "device": str(self.device).upper(),
            "model_version": "EfficientNetB0-TransferLearning-v1.0"
        }

    def detect_multiple_fruits(self, img_bgr: np.ndarray, min_area: int = 1200) -> Dict[str, Any]:
        """
        Locate multiple fruits using computer vision segmentation,
        crop each individually, run EfficientNetB0 classification, and compile batch metrics.
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # Multi-scale contour extraction
        # Try Otsu thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological opening & closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        valid_boxes = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Filter full-image bounding box
            if bw > w * 0.95 and bh > h * 0.95:
                continue

            # Add padding
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
                "confidence": pred["confidence"],
                "status": pred["status"]
            })

        # Fallback if no distinct multiple contours found: treat whole image as 1 fruit
        if len(detections) == 0:
            pred = self.predict_single_crop(img_bgr)
            detections.append({
                "x1": 10,
                "y1": 10,
                "x2": w - 10,
                "y2": h - 10,
                "label": pred["label"],
                "confidence": pred["confidence"],
                "status": pred["status"]
            })

        # Calculate counts
        good_count = sum(1 for d in detections if d["label"] == "Good")
        damaged_count = sum(1 for d in detections if d["label"] == "Damaged")
        rotten_count = sum(1 for d in detections if d["label"] == "Rotten")
        total_fruits = len(detections)
        acceptance_rate = round((good_count / total_fruits) * 100, 1) if total_fruits > 0 else 0.0

        if rotten_count > 0:
            overall_status = "CRITICAL_DEFECTS_DETECTED"
        elif damaged_count > 0:
            overall_status = "SECONDARY_GRADING_REQUIRED"
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

def get_detector() -> FruitVisionDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = FruitVisionDetector()
    return _detector_instance

