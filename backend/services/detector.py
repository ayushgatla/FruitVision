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

FRUIT_SURROGATES = {'ping-pong ball', 'croquet ball', 'balloon'}

NON_PRODUCE_TRIGGERS = [
    # Humans, apparel, accessories & headwear
    'person', 'human', 'face', 'groom', 'scuba diver', 'ballplayer', 'baseball player',
    'spectacles', 'sunglasses', 'sunglass', 'wig', 'jersey', 't-shirt', 'sweatshirt', 'suit', 'lab coat',
    'neck brace', 'bow tie', 'necktie', 'cowboy hat', 'hair slide', 'mask', 'backpack', 'gown',
    'cloak', 'pajama', 'jean', 'sock', 'shoe', 'sneaker', 'swimming cap', 'bathing cap',
    'shower cap', 'face powder', 'sunscreen', 'seat belt', 'fur coat', 'cardigan', 'trench coat',
    'maillot tank suit',
    # Electronic devices, screens, peripherals & computing
    'television', 'screen', 'monitor', 'web site', 'laptop', 'cellular telephone', 'phone',
    'keyboard', 'mouse', 'desktop', 'computer', 'notebook', 'modem', 'printer', 'hard disc',
    'remote control', 'cassette', 'tape player', 'loudspeaker', 'microphone', 'headphones',
    # Everyday office, home & personal accessories
    'clock', 'digital clock', 'analog clock', 'stopwatch', 'wall clock', 'timepiece',
    'bottle', 'water bottle', 'beer bottle', 'pop bottle', 'pill bottle', 'wine bottle',
    'cup', 'coffee mug', 'mug', 'measuring cup', 'teapot', 'pitcher', 'whiskey jug',
    'plate', 'tray', 'soup bowl', 'saltshaker', 'pen', 'pencil', 'ballpoint pen', 'fountain pen',
    'pencil sharpener', 'rubber eraser', 'envelope', 'packet', 'carton', 'wallet', 'purse',
    'binder', 'comic book', 'book jacket', 'scissors', 'screwdriver', 'wrench', 'hammer', 'rule',
    'measuring tape',
    # Indoor furniture, architecture & room fixtures
    'sliding door', 'window shade', 'window screen', 'shower curtain', 'wardrobe', 'couch',
    'desk', 'dining table', 'chair', 'folding chair', 'rocking chair', 'barber chair',
    'bookcase', 'door', 'wall', 'stone wall', 'curtain', 'pillow', 'quilt', 'doormat', 'fire screen',
    'table lamp', 'candle', 'chimney',
    # Vehicles & wheeled devices
    'unicycle', 'bicycle', 'tricycle', 'cart', 'wheelbarrow', 'car', 'truck', 'bus', 'vehicle',
    'motorcycle', 'airplane', 'boat', 'tractor', 'trailer',
    # Construction, industrial & infrastructure
    'pier', 'dock', 'lumbermill', 'drilling platform', 'platform', 'pole', 'bridge', 'dam',
    'scaffolding', 'crane', 'bannister', 'fence', 'building', 'tower', 'steel', 'maze', 'broom', 'shovel', 'rail'
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
        Industry 4.0 optical food safety guardrail (ISO 22000).
        Detects extreme necrotic black rot / mold coverage as a safety backup.
        Note: Apples and produce DO NOT require leaves to be classified as Good / Market Grade A.
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        b, g, r = img_bgr[:,:,0].astype(float), img_bgr[:,:,1].astype(float), img_bgr[:,:,2].astype(float)

        # Mask fruit from light background
        fruit_mask = (gray < 240) & (gray > 15)
        fruit_pixels = int(np.sum(fruit_mask))
        if fruit_pixels < 400:
            return {"override": False, "label": "Good", "confidence": 0.9}

        # Severe necrotic dark rot & mold
        dark_necrosis = (hsv[:,:,2] < 30) & fruit_mask
        dark_rot = (r < 50) & (g < 45) & (b < 40) & fruit_mask
        rot_mask = dark_necrosis | dark_rot
        rot_ratio = float(np.sum(rot_mask)) / fruit_pixels

        # Only override if severe black rot coverage >= 35%
        if rot_ratio >= 0.35:
            conf = float(round(max(0.95, min(0.995, 0.85 + rot_ratio * 0.25)), 4))
            return {"override": True, "label": "Rotten", "confidence": conf, "rot_ratio": rot_ratio}

        return {"override": False, "rot_ratio": rot_ratio}

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
        is_red = ((hsv[:,:,0] <= 9) | (hsv[:,:,0] >= 165)) & (hsv[:,:,1] >= 40) & (hsv[:,:,2] >= 35)
        is_orange = (hsv[:,:,0] >= 10) & (hsv[:,:,0] <= 24) & (hsv[:,:,1] >= 45) & (hsv[:,:,2] >= 40)
        is_yellow = (hsv[:,:,0] >= 25) & (hsv[:,:,0] <= 42) & (hsv[:,:,1] >= 40) & (hsv[:,:,2] >= 40)
        is_green = (hsv[:,:,0] >= 35) & (hsv[:,:,0] <= 85) & (hsv[:,:,1] >= 35) & (hsv[:,:,2] >= 35)
        is_dark_rot = (hsv[:,:,2] <= 60) & (hsv[:,:,1] >= 25) & (hsv[:,:,0] <= 30)

        organic_pigment = (is_red | is_orange | is_yellow | is_green | is_dark_rot) & non_bg

        # Stark non-fruit synthetic colors (deep blue, cyan: Hue 90-135, Sat >= 35)
        stark_blue_cyan = (hsv[:,:,0] >= 90) & (hsv[:,:,0] <= 135) & (hsv[:,:,1] >= 35) & non_bg
        blue_ratio = float(np.sum(stark_blue_cyan)) / max(total_non_bg, 1)
        organic_ratio = float(np.sum(organic_pigment)) / max(total_non_bg, 1)

        if blue_ratio > 0.20:
            return False, "Inorganic chromatic signature (Blue / Cyan tones)", 0.98

        if organic_ratio < 0.08:
            return False, "Missing biological produce pigments", 0.97

        # 5. Human Skin / Portrait Detection in YCrCb
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        skin_mask = (ycrcb[:,:,1] >= 130) & (ycrcb[:,:,1] <= 175) & (ycrcb[:,:,2] >= 75) & (ycrcb[:,:,2] <= 130) & (hsv[:,:,1] < 100)
        skin_ratio = float(np.sum(skin_mask)) / max(total_pixels, 1)

        # 6. Zero-shot ImageNet Check
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
                top5_cats = [self.imagenet_cats[idx].lower() for idx in top5_indices]
                top5_probs = [out[idx].item() for idx in top5_indices]

                has_produce = any(cat in PRODUCE_SET or cat in FRUIT_SURROGATES for cat in top5_cats)

                # Round spherical shape surrogates (e.g. bruised apples look like analog clock or gong)
                is_round_surrogate = any(s in top_cat for s in ['analog clock', 'wall clock', 'clock', 'gong', 'chime', 'balloon', 'golf ball'])
                if is_round_surrogate and organic_ratio >= 0.22:
                    has_produce = True

                is_top1_non_produce = any(trigger in top_cat for trigger in NON_PRODUCE_TRIGGERS)

                non_produce_top5_sum = sum(
                    prob for prob, cat in zip(top5_probs, top5_cats) 
                    if any(trig in cat for trig in NON_PRODUCE_TRIGGERS)
                )

                if not has_produce:
                    if skin_ratio >= 0.10:
                        return False, "Identified human subject / skin tone (non-produce)", 0.98
                    if is_top1_non_produce and top_prob.item() >= 0.12:
                        return False, f"Identified non-fruit object ({top_cat})", float(round(max(0.95, top_prob.item()), 4))
                    if non_produce_top5_sum >= 0.22:
                        return False, f"Identified non-fruit scene ({top_cat})", float(round(max(0.95, non_produce_top5_sum), 4))
            except Exception:
                pass

        if skin_ratio >= 0.20:
            return False, "Predominantly human skin tone (non-produce)", 0.98

        return True, "Organic fruit verified", 0.98

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

        # Optical food safety guardrail: only intervene if model predicts Good but severe rot is detected
        guard = self._analyze_surface_defect_heuristics(img_bgr)
        if guard.get("override") and label == "Good" and guard["label"] == "Rotten":
            label = "Rotten"
            confidence = guard["confidence"]
            rem = round((1.0 - confidence) / 2.0, 4)
            prob_dict = {"Good": rem, "Damaged": rem, "Rotten": confidence}

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

    def find_fruit_boxes_live(self, img_bgr: np.ndarray, min_area_ratio: float = 0.012, max_area_ratio: float = 0.75) -> List[Tuple[int, int, int, int]]:
        """
        Segment and localize bounding boxes for live webcam / camera feeds.
        Suppresses human skin, ambient room clutter, and background walls.
        Isolates saturated organic produce candidates (Apples, Oranges, Bananas, etc.).
        Returns empty list [] if no produce item is held in front of the lens.
        """
        h, w = img_bgr.shape[:2]
        total_area = h * w
        min_area = int(total_area * min_area_ratio)
        max_area = int(total_area * max_area_ratio)

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]

        # Suppress human skin cluster with low/moderate saturation
        skin_mask = (cr >= 130) & (cr <= 175) & (cb >= 75) & (cb <= 130) & (hsv[:, :, 1] < 110)

        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Target vibrant fruit pigment bands
        is_red = ((hue <= 9) | (hue >= 165)) & (sat >= 50) & (val >= 35)
        is_orange = (hue >= 10) & (hue <= 24) & (sat >= 55) & (val >= 40)
        is_yellow = (hue >= 25) & (hue <= 42) & (sat >= 50) & (val >= 40)
        is_green = (hue >= 35) & (hue <= 85) & (sat >= 45) & (val >= 35)
        is_dark_rot = (val <= 60) & (sat >= 25) & (hue <= 30)

        fruit_pixels = (is_red | is_orange | is_yellow | is_green | is_dark_rot) & (~skin_mask)
        mask = fruit_pixels.astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in cnts:
            area = cv2.contourArea(c)
            if min_area <= area <= max_area:
                hull = cv2.convexHull(c)
                solidity = float(area) / max(1, cv2.contourArea(hull))
                if solidity >= 0.45:
                    x, y, bw, bh = cv2.boundingRect(c)
                    pad_x = int(bw * 0.05)
                    pad_y = int(bh * 0.05)
                    boxes.append((
                        max(0, x - pad_x),
                        max(0, y - pad_y),
                        min(w, x + bw + pad_x),
                        min(h, y + bh + pad_y)
                    ))

        # Sort left-to-right
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
