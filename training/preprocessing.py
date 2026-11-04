"""
FruitVision AI — Preprocessing and Augmentation Pipeline
Consistent preprocessing for both training and inference.
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Inference / Validation transform
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Training data augmentation transform
train_transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

def preprocess_numpy_image(img_bgr: np.ndarray) -> torch.Tensor:
    """Preprocess an OpenCV BGR numpy array for EfficientNetB0 inference."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    tensor = inference_transform(pil_img)
    return tensor.unsqueeze(0)  # Shape: (1, 3, 224, 224)

def preprocess_pil_image(pil_img: Image.Image) -> torch.Tensor:
    """Preprocess a PIL image for inference."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    tensor = inference_transform(pil_img)
    return tensor.unsqueeze(0)

