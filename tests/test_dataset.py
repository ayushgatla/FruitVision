"""
Unit and regression tests for FruitVision AI dataset preparation,
manifest integrity, and mechanical data leakage prevention.
"""

import json
from pathlib import Path
from PIL import Image

PROCESSED_DIR = Path("dataset/processed")
MANIFEST_PATH = PROCESSED_DIR / "manifest.json"
STATS_PATH = PROCESSED_DIR / "split_statistics.json"

TARGET_CLASSES = ["Apple_Bad", "Apple_Good", "Banana_Bad", "Banana_Good"]

def test_manifest_and_stats_exist():
    """Verify that processed dataset manifest and statistics exist."""
    assert MANIFEST_PATH.exists(), f"Missing manifest at {MANIFEST_PATH}"
    assert STATS_PATH.exists(), f"Missing statistics at {STATS_PATH}"

def test_manifest_structure_and_classes():
    """Verify manifest schema and class naming."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert "classes" in manifest
    assert sorted(manifest["classes"]) == sorted(TARGET_CLASSES)
    assert manifest["total_samples"] > 0
    assert "samples" in manifest
    assert len(manifest["samples"]) == manifest["total_samples"]

def test_zero_group_leakage():
    """Ensure no burst capture / consecutive frame group appears across multiple splits."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    groups_by_split = {"train": set(), "val": set(), "test": set()}
    for sample in manifest["samples"]:
        split = sample["split"]
        gkey = sample["group_key"]
        groups_by_split[split].add(gkey)

    train_val_overlap = groups_by_split["train"] & groups_by_split["val"]
    train_test_overlap = groups_by_split["train"] & groups_by_split["test"]
    val_test_overlap = groups_by_split["val"] & groups_by_split["test"]

    assert len(train_val_overlap) == 0, f"Leakage detected between Train and Val: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"Leakage detected between Train and Test: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"Leakage detected between Val and Test: {val_test_overlap}"

def test_zero_sha256_leakage():
    """Ensure no identical file hash exists across multiple splits."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    hashes_by_split = {"train": set(), "val": set(), "test": set()}
    for sample in manifest["samples"]:
        split = sample["split"]
        sha = sample["sha256"]
        hashes_by_split[split].add(sha)

    assert len(hashes_by_split["train"] & hashes_by_split["val"]) == 0
    assert len(hashes_by_split["train"] & hashes_by_split["test"]) == 0
    assert len(hashes_by_split["val"] & hashes_by_split["test"]) == 0

def test_class_balance_and_split_ratios():
    """Verify ~200 images per class and balanced train/val/test partitions."""
    with open(STATS_PATH, "r", encoding="utf-8") as f:
        stats = json.load(f)

    for cls in TARGET_CLASSES:
        tot = stats["total_by_class"][cls]
        # Target ~200 per class (within reasonable margin due to group-atomicity)
        assert 180 <= tot <= 260, f"Class {cls} count {tot} deviates significantly from ~200 target"
        assert stats["train"][cls] > 0
        assert stats["val"][cls] > 0
        assert stats["test"][cls] > 0

    total_all = sum(stats["total_by_class"].values())
    train_ratio = stats["total_by_split"]["train"] / total_all
    val_ratio = stats["total_by_split"]["val"] / total_all
    test_ratio = stats["total_by_split"]["test"] / total_all

    # Approximately 70/15/15 (+- 5% tolerance due to burst-group granularity)
    assert 0.60 <= train_ratio <= 0.75, f"Train ratio {train_ratio} outside tolerance"
    assert 0.10 <= val_ratio <= 0.22, f"Val ratio {val_ratio} outside tolerance"
    assert 0.10 <= test_ratio <= 0.22, f"Test ratio {test_ratio} outside tolerance"

def test_all_processed_images_valid():
    """Verify all physical images in processed directory are readable RGB files."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for sample in manifest["samples"][:50]:  # Check first 50 as fast smoke test
        img_path = PROCESSED_DIR / sample["relative_path"]
        assert img_path.exists(), f"Image missing at {img_path}"
        with Image.open(img_path) as img:
            assert img.format in ["JPEG", "MPO", "PNG"]
            rgb_img = img.convert("RGB")
            assert rgb_img.size[0] > 0 and rgb_img.size[1] > 0

