"""
FruitVision AI — Real Fruit PyTorch Dataset and DataLoader Pipeline
Loads images from group-stratified processed splits (dataset/processed/)
with support for training data augmentation and inference normalization.
"""

import os
import json
from pathlib import Path
from typing import Optional, Callable, Tuple, List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from training.preprocessing import train_transform, inference_transform

CLASSES_4CLS = ["Apple_Bad", "Apple_Good", "Banana_Bad", "Banana_Good"]

class FruitDataset(Dataset):
    """
    Empirical PyTorch Dataset for FruitVision AI.
    Loads real images from prepared train/val/test splits.
    """
    def __init__(
        self,
        split: str = "train",
        dataset_root: str = "dataset/processed",
        transform: Optional[Callable] = None,
        classes: Optional[List[str]] = None
    ):
        self.split = split
        self.root = Path(dataset_root)
        self.transform = transform
        self.classes = classes or CLASSES_4CLS
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self):
        split_dir = self.root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}. "
                "Ensure `python scripts/prepare_dataset.py` has been executed."
            )

        for cls_name in self.classes:
            cls_folder = split_dir / cls_name
            if not cls_folder.exists():
                continue
            idx = self.class_to_idx[cls_name]
            image_files = sorted(
                list(cls_folder.glob("*.jpg")) +
                list(cls_folder.glob("*.jpeg")) +
                list(cls_folder.glob("*.png"))
            )
            for f in image_files:
                self.samples.append((f, idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid images found for split '{self.split}' in {split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")

        if self.transform is not None:
            tensor = self.transform(img)
        else:
            tensor = inference_transform(img)

        return tensor, label

    def get_class_counts(self) -> Dict[str, int]:
        counts = {cls_name: 0 for cls_name in self.classes}
        for _, label_idx in self.samples:
            cls_name = self.classes[label_idx]
            counts[cls_name] += 1
        return counts

    def compute_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for cross-entropy loss."""
        counts = self.get_class_counts()
        total = len(self.samples)
        num_classes = len(self.classes)
        weights = []
        for cls_name in self.classes:
            c = counts[cls_name]
            w = total / (num_classes * c) if c > 0 else 1.0
            weights.append(w)
        return torch.tensor(weights, dtype=torch.float32)

def create_dataloaders(
    dataset_root: str = "dataset/processed",
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = False,
    classes: Optional[List[str]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, FruitDataset]:
    """
    Creates reproducible DataLoaders for Train, Validation, and Test splits.
    """
    train_ds = FruitDataset(split="train", dataset_root=dataset_root, transform=train_transform, classes=classes)
    val_ds = FruitDataset(split="val", dataset_root=dataset_root, transform=inference_transform, classes=classes)
    test_ds = FruitDataset(split="test", dataset_root=dataset_root, transform=inference_transform, classes=classes)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader, train_ds

