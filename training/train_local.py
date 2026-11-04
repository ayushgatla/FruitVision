"""
FruitVision AI — Local EfficientNetB0 Model Training & Calibration Script
Trains transfer-learned EfficientNetB0 on three fruit quality classes:
0: Good
1: Damaged
2: Rotten
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm
from training.preprocessing import train_transform, inference_transform

CLASSES = ["Good", "Damaged", "Rotten"]

class SyntheticFruitDataset(Dataset):
    """
    Generates diverse fruit images (Apples, Oranges, Bananas, Lemons, Peaches)
    with realistic features for Good, Damaged, and Rotten classes,
    supplemented with disk sample images.
    """
    def __init__(self, size=600, transform=None):
        self.size = size
        self.transform = transform
        self.samples = []
        for i in range(size):
            label = i % 3
            self.samples.append(label)
        random.seed(42)
        random.shuffle(self.samples)

        # Cache disk sample images if available
        self.disk_samples = {0: [], 1: [], 2: []}
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        samples_dir = os.path.join(base_dir, "sample_images")
        if os.path.exists(samples_dir):
            if os.path.exists(os.path.join(samples_dir, "sample_good_apple.jpg")):
                self.disk_samples[0].append(Image.open(os.path.join(samples_dir, "sample_good_apple.jpg")).convert("RGB"))
            if os.path.exists(os.path.join(samples_dir, "sample_good_orange.jpg")):
                self.disk_samples[0].append(Image.open(os.path.join(samples_dir, "sample_good_orange.jpg")).convert("RGB"))
            if os.path.exists(os.path.join(samples_dir, "sample_good_banana.jpg")):
                self.disk_samples[0].append(Image.open(os.path.join(samples_dir, "sample_good_banana.jpg")).convert("RGB"))
            if os.path.exists(os.path.join(samples_dir, "sample_damaged_apple.jpg")):
                self.disk_samples[1].append(Image.open(os.path.join(samples_dir, "sample_damaged_apple.jpg")).convert("RGB"))
            if os.path.exists(os.path.join(samples_dir, "sample_rotten_apple.jpg")):
                self.disk_samples[2].append(Image.open(os.path.join(samples_dir, "sample_rotten_apple.jpg")).convert("RGB"))
            if os.path.exists(os.path.join(samples_dir, "sample_real_rotten_apple.jpg")):
                self.disk_samples[2].append(Image.open(os.path.join(samples_dir, "sample_real_rotten_apple.jpg")).convert("RGB"))
            if os.path.exists(os.path.join(samples_dir, "sample_rotten_banana.jpg")):
                self.disk_samples[2].append(Image.open(os.path.join(samples_dir, "sample_rotten_banana.jpg")).convert("RGB"))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        label = self.samples[idx]
        # In 60% of cases, use an augmented real disk sample if available
        if self.disk_samples[label] and random.random() < 0.60:
            img = random.choice(self.disk_samples[label]).copy()
        else:
            img = self._generate_fruit_image(label)
        if self.transform:
            img = self.transform(img)
        return img, label

    def _generate_fruit_image(self, label: int) -> Image.Image:
        img = Image.new("RGB", (256, 256), color=(245, 245, 245))
        draw = ImageDraw.Draw(img)

        fruit_type = random.choice(["apple", "orange", "banana", "pear", "peach"])
        
        # Color palette per fruit
        if fruit_type == "apple":
            base_col = (random.randint(180, 230), random.randint(20, 50), random.randint(30, 60))
            bbox = (48, 55, 208, 215)
        elif fruit_type == "orange":
            base_col = (random.randint(235, 255), random.randint(110, 160), random.randint(10, 30))
            bbox = (50, 50, 206, 206)
        elif fruit_type == "banana":
            # Bright yellow for Good bananas, dark rot for Rotten
            if label == 0:
                base_col = (random.randint(240, 255), random.randint(215, 240), random.randint(15, 45))
            elif label == 1:
                base_col = (random.randint(210, 235), random.randint(180, 210), random.randint(20, 50))
            else:
                base_col = (random.randint(40, 70), random.randint(35, 60), random.randint(25, 45))
            bbox = (40, 80, 216, 176)
        elif fruit_type == "pear":
            base_col = (random.randint(160, 200), random.randint(190, 220), random.randint(40, 70))
            bbox = (55, 60, 200, 220)
        else: # peach
            base_col = (random.randint(240, 255), random.randint(140, 180), random.randint(100, 140))
            bbox = (50, 50, 206, 206)

        # Draw base fruit
        if fruit_type == "banana":
            draw.chord(bbox, start=30, end=190, fill=base_col, outline=None)
        else:
            draw.ellipse(bbox, fill=base_col, outline=None)

        # Highlight for Good fruits
        if label == 0:  # Good (Fresh)
            hl_x = random.randint(70, 100)
            hl_y = random.randint(70, 100)
            draw.ellipse((hl_x, hl_y, hl_x + 30, hl_y + 18), fill=(255, 255, 255, 120))
            # Green stem / leaf
            draw.line((128, 52, 125, 30), fill=(40, 80, 30), width=4)
            draw.polygon([(125, 35), (150, 20), (145, 45)], fill=(30, 160, 40))

        elif label == 1:  # Damaged (Bruised / Cuts / Scratches)
            # Impact bruise
            br_x = random.randint(100, 160)
            br_y = random.randint(100, 160)
            bruise_col = (max(0, base_col[0] - 100), max(0, base_col[1] - 60), max(0, base_col[2] - 30))
            draw.ellipse((br_x - 25, br_y - 20, br_x + 25, br_y + 20), fill=bruise_col)
            # Surface cuts or scars
            cut_x1 = random.randint(80, 150)
            cut_y1 = random.randint(90, 150)
            draw.line((cut_x1, cut_y1, cut_x1 + 35, cut_y1 + 25), fill=(40, 20, 10), width=3)
            draw.line((cut_x1 + 5, cut_y1 + 10, cut_x1 + 20, cut_y1 + 35), fill=(50, 25, 15), width=2)

        elif label == 2:  # Rotten (Decay, Mold, Sunken Patches)
            # Extensive dark brown / black sunken rot
            rot_x = random.randint(90, 140)
            rot_y = random.randint(90, 150)
            rot_col = (random.randint(25, 55), random.randint(20, 45), random.randint(15, 35))
            draw.ellipse((rot_x - 45, rot_y - 40, rot_x + 45, rot_y + 40), fill=rot_col)
            # Gray/white/green fungal mold colonies
            for _ in range(40):
                mx = rot_x + random.randint(-35, 35)
                my = rot_y + random.randint(-30, 30)
                mold_col = random.choice([(180, 195, 185), (140, 165, 145), (70, 100, 80)])
                draw.ellipse((mx - 3, my - 3, mx + 3, my + 3), fill=mold_col)

        return img



def build_model(num_classes=3):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    # Freeze backbone initially
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Custom classification head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(128, num_classes)
    )
    return model

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FruitVision AI] Training device: {device}")

    # Dataset splits
    train_dataset = SyntheticFruitDataset(size=180, transform=train_transform)
    val_dataset = SyntheticFruitDataset(size=60, transform=inference_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = build_model(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    print("[FruitVision AI] Step 1: Training Classification Head (Backbone Frozen)...")
    epochs_head = 3
    for epoch in range(1, epochs_head + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        print(f"Epoch {epoch}/{epochs_head} - Loss: {total_loss/total:.4f} - Accuracy: {train_acc*100:.2f}%")

    print("\n[FruitVision AI] Step 2: Fine-Tuning Top Convolutional Blocks (LR=1e-4)...")
    # Unfreeze top feature layers (stages 6 and 7)
    for param in model.features[6:].parameters():
        param.requires_grad = True

    optimizer = optim.Adam([
        {'params': model.features[6:].parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 2e-4}
    ])

    best_val_acc = 0.0
    epochs_ft = 3
    weights_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
    os.makedirs(weights_dir, exist_ok=True)
    best_path = os.path.join(weights_dir, "best.pt")
    efficientnet_path = os.path.join(weights_dir, "fruit_efficientnet_b0.pt")

    for epoch in range(1, epochs_ft + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Fine-Tuning Epoch {epoch}/{epochs_ft} - Val Loss: {val_loss/val_total:.4f} - Val Accuracy: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': CLASSES,
                'val_accuracy': val_acc,
                'architecture': 'EfficientNetB0'
            }, best_path)
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': CLASSES,
                'val_accuracy': val_acc,
                'architecture': 'EfficientNetB0'
            }, efficientnet_path)
            print(f"  --> Saved new best model checkpoint with accuracy: {val_acc*100:.2f}%")

    print(f"\n[FruitVision AI] Training Complete! Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"[FruitVision AI] Saved to: {best_path} and {efficientnet_path}")

if __name__ == "__main__":
    train()
