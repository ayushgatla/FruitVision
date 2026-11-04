"""
FruitVision AI — Reproducible Dataset Preparation & Group-Aware Stratified Splitting
Prepares an empirical ~800-image dataset (4 classes: Apple_Good, Apple_Bad, Banana_Good, Banana_Bad)
with strict group-aware splitting to prevent data leakage from consecutive/burst captures.

Outputs:
- dataset/processed/train/<class>/
- dataset/processed/val/<class>/
- dataset/processed/test/<class>/
- dataset/processed/manifest.json
- dataset/processed/split_statistics.json
- dataset/splits/split_manifest.json
"""

import os
import re
import json
import shutil
import hashlib
import random
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any
from PIL import Image

TARGET_CLASSES = ["Apple_Bad", "Apple_Good", "Banana_Bad", "Banana_Good"]
DEFAULT_TARGET_PER_CLASS = 200
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42

def extract_image_group_key(filename: str, class_name: str) -> str:
    """
    Extract a grouping key for an image based on burst capture timestamps,
    shutter index ranges, or repeated suffixes (_01, _02).
    Images of the same fruit capture session share the same group key.
    """
    stem = Path(filename).stem

    # Pattern A: Timestamp IMGYYYYMMDDHHMMSS or YYYYMMDD_HHMMSS
    # e.g., IMG20200728180147 or 20190809_115439
    ts_match = re.search(r'(?:IMG)?(20\d{6})[_\-]?(\d{6})', stem)
    if ts_match:
        date_str, time_str = ts_match.group(1), ts_match.group(2)
        try:
            dt = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
            # Cluster into 1-minute (60-second) burst windows
            minute_bucket = dt.strftime("%Y%m%d_%H%M")
            return f"{class_name}_burst_{minute_bucket}"
        except ValueError:
            pass

    # Pattern B: Sequential camera indices (e.g. IMG_9646)
    idx_match = re.search(r'IMG[_\-](\d+)', stem, re.IGNORECASE)
    if idx_match:
        idx_val = int(idx_match.group(1))
        # Cluster indices into buckets of 4 (e.g. 9640-9643)
        idx_bucket = idx_val // 4
        return f"{class_name}_idx_{idx_bucket}"

    # Pattern C: Repeated suffix like _01, _02
    clean_stem = re.sub(r'_\d+$', '', stem)
    return f"{class_name}_stem_{clean_stem}"

def verify_image(path: Path) -> bool:
    """Validate image file readability and RGB conversion."""
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.convert("RGB")
        return True
    except Exception:
        return False

def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash of file."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def prepare_dataset(
    source_dir: str = "dataset",
    output_dir: str = "dataset/processed",
    splits_dir: str = "dataset/splits",
    target_classes: List[str] = TARGET_CLASSES,
    samples_per_class: int = DEFAULT_TARGET_PER_CLASS,
    seed: int = RANDOM_SEED
) -> Dict[str, Any]:
    random.seed(seed)
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    splits_path = Path(splits_dir)

    output_path.mkdir(parents=True, exist_ok=True)
    splits_path.mkdir(parents=True, exist_ok=True)

    print(f"[Dataset Prep] Source directory: {source_path.resolve()}")
    print(f"[Dataset Prep] Target classes: {target_classes}")
    print(f"[Dataset Prep] Target samples per class: {samples_per_class}")
    print(f"[Dataset Prep] Seed: {seed}")

    all_groups_by_class: Dict[str, Dict[str, List[Path]]] = {}
    verified_files_by_class: Dict[str, List[Path]] = {}
    corrupted_count = 0

    for cls in target_classes:
        cls_dir = source_path / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {cls_dir}")

        files = sorted(list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpeg")))
        valid_files = []
        groups = defaultdict(list)

        for f in files:
            if verify_image(f):
                valid_files.append(f)
                gkey = extract_image_group_key(f.name, cls)
                groups[gkey].append(f)
            else:
                corrupted_count += 1

        verified_files_by_class[cls] = valid_files
        all_groups_by_class[cls] = dict(groups)
        print(f"  Class '{cls}': {len(valid_files)} valid images across {len(groups)} distinct burst/capture groups.")

    # Select ~samples_per_class images per class by sampling whole groups
    selected_samples: Dict[str, Dict[str, List[Path]]] = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}
    assigned_groups: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}

    train_ratio = SPLIT_RATIOS["train"]
    val_ratio = SPLIT_RATIOS["val"]
    test_ratio = SPLIT_RATIOS["test"]

    for cls in target_classes:
        groups = list(all_groups_by_class[cls].items())
        random.shuffle(groups)

        target_train = int(samples_per_class * train_ratio)
        target_val = int(samples_per_class * val_ratio)
        target_test = samples_per_class - target_train - target_val

        cls_train, cls_val, cls_test = [], [], []

        for gkey, gfiles in groups:
            # Decide split assignment prioritizing train, then val, then test
            if len(cls_train) < target_train:
                cls_train.extend(gfiles)
                assigned_groups["train"].add(gkey)
            elif len(cls_val) < target_val:
                cls_val.extend(gfiles)
                assigned_groups["val"].add(gkey)
            elif len(cls_test) < target_test:
                cls_test.extend(gfiles)
                assigned_groups["test"].add(gkey)
            else:
                # Stop once all target split quotas are satisfied
                break

        selected_samples["train"][cls] = cls_train
        selected_samples["val"][cls] = cls_val
        selected_samples["test"][cls] = cls_test

    # Mechanical Data Leakage Verification
    train_groups = assigned_groups["train"]
    val_groups = assigned_groups["val"]
    test_groups = assigned_groups["test"]

    leakage_train_val = train_groups & val_groups
    leakage_train_test = train_groups & test_groups
    leakage_val_test = val_groups & test_groups

    if leakage_train_val or leakage_train_test or leakage_val_test:
        raise ValueError(
            f"DATA LEAKAGE DETECTED! "
            f"Train/Val overlap: {len(leakage_train_val)}, "
            f"Train/Test overlap: {len(leakage_train_test)}, "
            f"Val/Test overlap: {len(leakage_val_test)}"
        )

    print("[Dataset Prep] Mechanical assertion PASSED: 0 group overlap between Train, Val, and Test splits.")

    # Clean and recreate processed directory structure
    for split in ["train", "val", "test"]:
        split_dir = output_path / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        for cls in target_classes:
            (split_dir / cls).mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    sha256_by_split = defaultdict(set)
    stats_summary = {"train": {}, "val": {}, "test": {}, "total_by_split": {}, "total_by_class": {}}

    for split in ["train", "val", "test"]:
        total_split = 0
        for cls in target_classes:
            file_list = selected_samples[split][cls]
            stats_summary[split][cls] = len(file_list)
            total_split += len(file_list)

            for src_file in file_list:
                dest_file = output_path / split / cls / src_file.name
                shutil.copy2(src_file, dest_file)

                f_hash = compute_sha256(src_file)
                sha256_by_split[split].add(f_hash)

                rel_dest = dest_file.relative_to(output_path).as_posix()
                manifest_entries.append({
                    "relative_path": rel_dest,
                    "filename": src_file.name,
                    "class": cls,
                    "split": split,
                    "group_key": extract_image_group_key(src_file.name, cls),
                    "sha256": f_hash,
                    "source_path": src_file.as_posix()
                })
        stats_summary["total_by_split"][split] = total_split

    for cls in target_classes:
        stats_summary["total_by_class"][cls] = (
            stats_summary["train"][cls] + stats_summary["val"][cls] + stats_summary["test"][cls]
        )

    # Hash leakage assertion
    hash_leak_train_val = sha256_by_split["train"] & sha256_by_split["val"]
    hash_leak_train_test = sha256_by_split["train"] & sha256_by_split["test"]
    hash_leak_val_test = sha256_by_split["val"] & sha256_by_split["test"]
    assert len(hash_leak_train_val) == 0, f"Hash leak Train-Val: {len(hash_leak_train_val)}"
    assert len(hash_leak_train_test) == 0, f"Hash leak Train-Test: {len(hash_leak_train_test)}"
    assert len(hash_leak_val_test) == 0, f"Hash leak Val-Test: {len(hash_leak_val_test)}"

    # Save manifest and statistics
    manifest_data = {
        "classes": target_classes,
        "total_samples": len(manifest_entries),
        "split_counts": stats_summary["total_by_split"],
        "class_counts": stats_summary["total_by_class"],
        "samples": manifest_entries
    }

    manifest_file = output_path / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    stats_file = output_path / "split_statistics.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats_summary, f, indent=2)

    # Save lightweight split manifest for checkpoints & test loaders
    splits_manifest_file = splits_path / "split_manifest.json"
    with open(splits_manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print("\n" + "="*60)
    print("FRUITVISION AI — DATASET PREPARATION SUMMARY")
    print("="*60)
    for cls in target_classes:
        tr = stats_summary["train"][cls]
        va = stats_summary["val"][cls]
        te = stats_summary["test"][cls]
        tot = stats_summary["total_by_class"][cls]
        print(f"  {cls:<15} -> Train: {tr:3d} | Val: {va:3d} | Test: {te:3d} | Total: {tot:3d}")
    print("-"*60)
    print(f"  {'TOTAL':<15} -> Train: {stats_summary['total_by_split']['train']:3d} | Val: {stats_summary['total_by_split']['val']:3d} | Test: {stats_summary['total_by_split']['test']:3d} | Total: {len(manifest_entries):3d}")
    print("="*60)
    print(f"[Dataset Prep] Output stored in: {output_path.resolve()}")
    print(f"[Dataset Prep] Manifest saved to: {manifest_file.resolve()}")
    print(f"[Dataset Prep] Split stats saved to: {stats_file.resolve()}")

    return manifest_data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FruitVision AI Dataset Preparer")
    parser.add_argument("--source", type=str, default="dataset", help="Raw dataset root")
    parser.add_argument("--output", type=str, default="dataset/processed", help="Output directory")
    parser.add_argument("--samples-per-class", type=int, default=DEFAULT_TARGET_PER_CLASS, help="Samples target per class")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    args = parser.parse_args()

    prepare_dataset(
        source_dir=args.source,
        output_dir=args.output,
        samples_per_class=args.samples_per_class,
        seed=args.seed
    )

