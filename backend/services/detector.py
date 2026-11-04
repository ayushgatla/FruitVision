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
    # Vehicles & wheeled devices
    'unicycle', 'bicycle', 'tricycle', 'cart', 'wheelbarrow', 'car', 'truck', 'bus', 'vehicle',
    'motorcycle', 'airplane', 'boat', 'tractor', 'trailer',
    # Construction, industrial & architecture
    'pier', 'dock', 'lumbermill', 'drilling platform', 'platform', 'pole', 'bridge', 'dam',
    'scaffolding', 'crane', 'bannister', 'fence', 'building', 'tower', 'steel', 'maze',
    'broom', 'shovel', 'rail', 'chimney', 'monolith', 'megalith',
    # Electronics, screens & devices
    'television', 'screen', 'monitor', 'web site', 'laptop', 'cellular telephone', 'phone',
    'keyboard', 'mouse', 'desktop', 'computer',
    # Humans, apparel, accessories & headwear
    'sunglass', 'sunglasses', 'wig', 'shower cap', 'lab coat', 'sweatshirt', 'suit',
    'jersey', 'trench coat', 'neck brace', 'bow tie', 'necktie', 'cowboy hat', 'hair slide',
    'mask', 'backpack', 't-shirt', 'gown', 'cloak', 'pajama', 'jean', 'sock', 'shoe', 'sneaker',
    'spectacles', 'glasses', 'swimming cap', 'bathing cap',
    # Indoor / Furniture / Room
    'sliding door', 'window shade', 'window screen', 'shower curtain', 'wardrobe', 'couch',
    'desk', 'dining table', 'chair', 'bookcase', 'door', 'wall', 'room', 'curtain', 'pillow',
    # Animals
    'dog', 'cat', 'horse', 'bird',
    # Everyday non-produce tools & objects
    'clock', 'digital clock', 'analog clock', 'stopwatch', 'bottle', 'cup', 'mug',
    'pen', 'pencil', 'scissors', 'screwdriver', 'envelope', 'packet', 'wallet'
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
            base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            in_features = base_model.classifier[1].in_features
            base_model.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(128, len(self.classes))
            )

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

        # Case A: Overwhelmingly fresh, vibrant peel with low rot (< 8%)
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

        # 1. Non-background mask
        non_bg = (gray < 242) & (gray > 12)
        total_non_bg = int(np.sum(non_bg))

        if total_non_bg < 350:
            return False, "Blank image or background only", 0.99

        # 2. Text / Code screenshot detection
        edges = cv2.Canny(gray, 100, 200)
        edge_ratio = float(np.sum(edges > 0)) / total_pixels
        if edge_ratio > 0.16 and total_non_bg > 0.75 * total_pixels:
            return False, "Document or text screenshot detected", 0.98

        # 3. Full-Frame Scene / Landscape Detection
        border_pixels = np.concatenate([
            gray[0:max(1, int(h*0.05)), :].flatten(),
            gray[-max(1, int(h*0.05)):, :].flatten(),
            gray[:, 0:max(1, int(w*0.05))].flatten(),
            gray[:, -max(1, int(w*0.05)):].flatten()
        ])
        border_std = float(np.std(border_pixels))
        top_pixels = gray[0:max(1, int(h*0.1)), :].flatten()
        bottom_pixels = gray[-max(1, int(h*0.1)):, :].flatten()
        top_bottom_diff = abs(float(np.mean(top_pixels)) - float(np.mean(bottom_pixels)))

        # Outdoor blue sky detection in upper 40% of image
        top_hsv = hsv[:int(h*0.4), :]
        sky_mask = (top_hsv[:,:,0] >= 95) & (top_hsv[:,:,0] <= 130) & (top_hsv[:,:,1] >= 20) & (top_hsv[:,:,2] >= 80)
        sky_ratio = float(np.sum(sky_mask)) / max(1, top_hsv.shape[0] * top_hsv.shape[1])

        if sky_ratio > 0.15 and border_std > 30.0:
            return False, "Outdoor landscape / sky scene detected (non-produce)", 0.98

        if border_std > 45.0 and top_bottom_diff > 50.0:
            return False, "Complex outdoor / industrial scene detected (non-produce)", 0.97

        # 4. Organic fruit color ranges:
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

        # 5. Zero-shot ImageNet Check
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
                is_top1_non_produce = any(trigger in top_cat for trigger in NON_PRODUCE_TRIGGERS)

                non_produce_top5_sum = sum(
                    out[idx].item() for idx in top5_indices 
                    if any(trig in self.imagenet_cats[idx].lower() for trig in NON_PRODUCE_TRIGGERS)
                )

                if not has_produce_in_top5:
                    if is_top1_non_produce and top_prob.item() >= 0.10:
                        return False, f"Identified non-fruit object ({top_cat})", float(round(max(0.95, top_prob.item()), 4))
                    if non_produce_top5_sum >= 0.25:
                        return False, f"Identified non-fruit scene ({top_cat})", float(round(max(0.95, non_produce_top5_sum), 4))
            except Exception:
                pass

        return True, "Organic fruit verified", 0.98

    def predict_single_crop(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """Perform EfficientNetB0 inference on an individual fruit crop."""
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

        prob_dict = {
            self.classes[i]: float(round(probs[i], 4))
            for i in range(len(self.classes))
        }

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

    def find_fruit_boxes(self, img_bgr: np.ndarray, min_area_ratio: float = 0.012) -> List[Tuple[int, int, int, int]]:
        """
        Segment and localize bounding boxes for all distinct fruit instances on screen.
        Handles light (white bench) and dark (conveyor) backgrounds with color saliency.
        """
        h, w = img_bgr.shape[:2]
        total_area = h * w
        min_area = int(total_area * min_area_ratio)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # 1. Border sampling for background luminance estimation
        border_h = max(1, int(h * 0.05))
        border_w = max(1, int(w * 0.05))
        borders = np.concatenate([
            gray[:border_h, :].flatten(),
            gray[-border_h:, :].flatten(),
            gray[:, :border_w].flatten(),
            gray[:, -border_w:].flatten()
        ])
        bg_is_light = np.median(borders) > 128

        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        if bg_is_light:
            mask = ((gray < 225) | (sat > 35)) & (val > 15)
        else:
            mask = ((gray > 40) | (sat > 35)) & (val > 25)

        # Organic fruit hue detection
        fruit_hues = (
            ((hsv[:, :, 0] >= 0) & (hsv[:, :, 0] <= 85)) |
            ((hsv[:, :, 0] >= 160) & (hsv[:, :, 0] <= 179))
        )
        active_mask = (mask & fruit_hues & (sat > 25)).astype(np.uint8) * 255
        if np.sum(active_mask > 0) < min_area:
            active_mask = mask.astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        active_mask = cv2.morphologyEx(active_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        active_mask = cv2.morphologyEx(active_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(active_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Avoid full frame box if multiple items are present
            if bw > w * 0.96 and bh > h * 0.96:
                continue
            pad_x = int(bw * 0.04)
            pad_y = int(bh * 0.04)
            boxes.append((
                max(0, x - pad_x),
                max(0, y - pad_y),
                min(w, x + bw + pad_x),
                min(h, y + bh + pad_y)
            ))

        # Fallback if no separate boxes found: check if there's one large contour
        if not boxes:
            all_cnts, _ = cv2.findContours(active_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if all_cnts:
                largest = max(all_cnts, key=cv2.contourArea)
                lx, ly, lbw, lbh = cv2.boundingRect(largest)
                boxes.append((
                    max(0, lx),
                    max(0, ly),
                    min(w, lx + lbw),
                    min(h, ly + lbh)
                ))
            else:
                boxes.append((8, 8, w - 8, h - 8))

        # Sort boxes left-to-right
        boxes.sort(key=lambda b: b[0])
        return boxes

    def detect_multiple_fruits(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Locate all fruits using computer vision segmentation, crop each individually,
        run EfficientNetB0 classification on each, compile batch aggregate metrics,
        and render visible bounding boxes on the evidence image.
        """
        boxes = self.find_fruit_boxes(img_bgr)
        detections = []

        for idx, (x1, y1, x2, y2) in enumerate(boxes):
            crop = img_bgr[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                crop = img_bgr

            pred = self.predict_single_crop(crop)
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

        # Calculate batch statistics
        total_fruits = len(detections)
        good_count = sum(1 for d in detections if d["label"] == "Good")
        damaged_count = sum(1 for d in detections if d["label"] == "Damaged")
        rotten_count = sum(1 for d in detections if d["label"] == "Rotten")
        not_fruit_count = sum(1 for d in detections if d["label"] in ["Not Fruits", "Not Fruit"])

        acceptance_rate = round((good_count / total_fruits) * 100, 1) if total_fruits > 0 else 0.0

        # Classify all together: aggregate sorting decision & primary prediction
        if rotten_count > 0:
            overall_status = "CRITICAL_DEFECTS_DETECTED"
            primary_pred = "Rotten"
            rotten_confs = [d["confidence"] for d in detections if d["label"] == "Rotten"]
            agg_conf = max(rotten_confs) if rotten_confs else 0.95
            suggested_action = "Immediate Diverter Ejection to Bio-Waste / Composting"
        elif damaged_count > 0:
            overall_status = "SECONDARY_GRADING_REQUIRED"
            primary_pred = "Damaged"
            damaged_confs = [d["confidence"] for d in detections if d["label"] == "Damaged"]
            agg_conf = max(damaged_confs) if damaged_confs else 0.90
            suggested_action = "Route to Secondary Manual Grading / Reprocessing Unit"
        elif not_fruit_count == total_fruits:
            overall_status = "NON_PRODUCE_REJECTED"
            primary_pred = "Not Fruits"
            agg_conf = 0.98
            suggested_action = "Rejected: Foreign Non-Produce Identified"
        else:
            overall_status = "BATCH_ACCEPTED"
            primary_pred = "Good"
            good_confs = [d["confidence"] for d in detections if d["label"] == "Good"]
            agg_conf = float(round(sum(good_confs) / len(good_confs), 4)) if good_confs else 0.95
            suggested_action = "Direct to Automated Packaging Line (Market Grade A)"

        # Aggregate probability breakdown
        if total_fruits > 0:
            agg_probs = {
                "Good": round(good_count / total_fruits, 4),
                "Damaged": round(damaged_count / total_fruits, 4),
                "Rotten": round(rotten_count / total_fruits, 4)
            }
        else:
            agg_probs = {"Good": 1.0, "Damaged": 0.0, "Rotten": 0.0}

        # Draw visible bounding boxes on the saved evidence image
        annotated = draw_multi_annotations(img_bgr, detections)
        _, evidence_url = save_evidence_snapshot(annotated, prefix="multi_batch")

        return {
            "total_fruits": total_fruits,
            "good_count": good_count,
            "damaged_count": damaged_count,
            "rotten_count": rotten_count,
            "acceptance_rate": acceptance_rate,
            "overall_status": overall_status,
            "status": overall_status,
            "predicted_class": primary_pred,
            "confidence": agg_conf,
            "suggested_action": suggested_action,
            "probabilities": agg_probs,
            "detections": detections,
            "annotated_image_url": evidence_url
        }

    def detect_and_classify_image(self, img_bgr: np.ndarray, source_id: str = "upload") -> Dict[str, Any]:
        """
        Inspect image. Automatically detects whether single or multiple fruits appear on screen.
        If multiple fruits are detected, identifies each, classifies them separately and together,
        and renders visible bounding boxes around each fruit.
        """
        boxes = self.find_fruit_boxes(img_bgr)
        h, w = img_bgr.shape[:2]

        if len(boxes) > 1:
            # Multi-fruit on screen
            multi_res = self.detect_multiple_fruits(img_bgr)
            return {
                "is_multi": True,
                "fruit_count": multi_res["total_fruits"],
                "total_fruits": multi_res["total_fruits"],
                "good_count": multi_res["good_count"],
                "damaged_count": multi_res["damaged_count"],
                "rotten_count": multi_res["rotten_count"],
                "acceptance_rate": multi_res["acceptance_rate"],
                "predicted_class": multi_res["predicted_class"],
                "confidence": multi_res["confidence"],
                "status": multi_res["overall_status"],
                "suggested_action": multi_res["suggested_action"],
                "probabilities": multi_res["probabilities"],
                "detections": multi_res["detections"],
                "annotated_image_url": multi_res["annotated_image_url"],
                "device": str(self.device).upper(),
                "model_version": "EfficientNetB0-TransferLearning-v1.0"
            }

        # Single fruit on screen
        box = boxes[0] if len(boxes) == 1 else (8, 8, w - 8, h - 8)
        crop = img_bgr[box[1]:box[3], box[0]:box[2]]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            crop = img_bgr

        pred = self.predict_single_crop(crop)
        annotated = draw_single_annotation(img_bgr, pred["label"], pred["confidence"], box=box)
        _, evidence_url = save_evidence_snapshot(annotated, prefix=f"single_{pred['label'].lower().replace(' ', '_')}")

        detection_item = {
            "x1": int(box[0]),
            "y1": int(box[1]),
            "x2": int(box[2]),
            "y2": int(box[3]),
            "label": pred["label"],
            "confidence": pred["confidence"],
            "status": pred["status"]
        }

        return {
            "is_multi": False,
            "fruit_count": 1,
            "total_fruits": 1,
            "good_count": 1 if pred["label"] == "Good" else 0,
            "damaged_count": 1 if pred["label"] == "Damaged" else 0,
            "rotten_count": 1 if pred["label"] == "Rotten" else 0,
            "acceptance_rate": 100.0 if pred["label"] == "Good" else 0.0,
            "predicted_class": pred["label"],
            "confidence": pred["confidence"],
            "status": pred["status"],
            "suggested_action": pred["suggested_action"],
            "probabilities": pred["probabilities"],
            "detections": [detection_item],
            "annotated_image_url": evidence_url,
            "device": str(self.device).upper(),
            "model_version": "EfficientNetB0-TransferLearning-v1.0"
        }

_detector_instance: Optional[FruitVisionDetector] = None

def get_detector() -> FruitVisionDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = FruitVisionDetector()
    return _detector_instance
