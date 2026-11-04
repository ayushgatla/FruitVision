"""
FruitVision AI — Dataset Preparation and Harmonization Pipeline
Prepares unified 3-class dataset: Good, Damaged, Rotten
from public sources (Fruits-360 + Fruit Defect Datasets).
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image

def verify_and_clean_image(image_path: Path) -> bool:
    """Verify if image file is readable and non-corrupted."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        # Ensure it can actually be loaded and converted to RGB
        with Image.open(image_path) as img:
            img.convert("RGB")
        return True
    except Exception:
        return False

def organize_dataset_splits(source_dir: str, output_dir: str, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Split unified images into train, val, and test partitions
    with stratified distribution across Good, Damaged, and Rotten classes.
    """
    classes = ["Good", "Damaged", "Rotten"]
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    for split in ["train", "val", "test"]:
        for cls in classes:
            (output_path / split / cls).mkdir(parents=True, exist_ok=True)

    summary = {cls: {"train": 0, "val": 0, "test": 0, "corrupted": 0} for cls in classes}

    for cls in classes:
        cls_dir = source_path / cls
        if not cls_dir.exists():
            print(f"Directory for class '{cls}' not found at {cls_dir}. Skipping.")
            continue

        images = [p for p in cls_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
        random.seed(42)
        random.shuffle(images)

        valid_images = []
        for img_path in images:
            if verify_and_clean_image(img_path):
                valid_images.append(img_path)
            else:
                summary[cls]["corrupted"] += 1

        n_total = len(valid_images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        splits = {
            "train": valid_images[:n_train],
            "val": valid_images[n_train:n_train + n_val],
            "test": valid_images[n_train + n_val:]
        }

        for split_name, split_files in splits.items():
            for f in split_files:
                dest = output_path / split_name / cls / f.name
                shutil.copy2(f, dest)
                summary[cls][split_name] += 1

    print("\n=== Dataset Harmonization Summary ===")
    for cls in classes:
        print(f"Class: {cls.upper():<8} -> Train: {summary[cls]['train']} | Val: {summary[cls]['val']} | Test: {summary[cls]['test']} | Corrupted Discarded: {summary[cls]['corrupted']}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FruitVision AI Dataset Preparer")
    parser.add_argument("--source", type=str, default="raw_dataset", help="Raw images directory")
    parser.add_argument("--output", type=str, default="dataset", help="Prepared dataset directory")
    args = parser.parse_args()

    organize_dataset_splits(args.source, args.output)

