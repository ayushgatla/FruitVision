# FruitVision AI — Dataset Guide

## 1. Overview & Source
The real fruit dataset used in FruitVision AI consists of **6,654 real organic fruit images** organized into 6 classes:
- `Apple_Bad`: 1,141 images
- `Apple_Good`: 1,134 images
- `Banana_Bad`: 1,087 images
- `Banana_Good`: 1,113 images
- `Lime_Bad`: 1,085 images
- `Lime_Good`: 1,094 images

### License & Provenance
Public agricultural benchmark dataset capturing real produce specimens under variable illumination, orientation, and defect stages (fresh Grade A produce vs necrotic decay, mold, and impact bruising).

---

## 2. Empirical Dataset Audit Summary

An automated audit was conducted using `scripts/audit_dataset.py` (persisted in `reports/dataset_audit.json` and `reports/DATASET_AUDIT.md`):
- **Total Scanned Images:** 6,654
- **Valid & Readable (PIL verify + RGB convert):** 6,654 (100.0%)
- **Corrupted / Unreadable Images:** 0
- **Exact Byte Duplicates (SHA-256):** 0
- **Common Resolutions:**
  - 256 × 256: 3,490 images (52.45%)
  - 192 × 256: 2,260 images (33.96%)
  - 256 × 192: 883 images (13.27%)
  - 144 × 256: 12 images (0.18%)
  - 256 × 144: 9 images (0.14%)
- **Class Imbalance:** Maximum deviation between classes is < 3% (1.05:1 ratio), providing a naturally balanced multi-class distribution.

---

## 3. Data Leakage Prevention & Group-Aware Splitting

### The Burst-Capture Problem
Analysis revealed that images in the raw dataset were captured in burst sequences (frames taken 1–5 seconds apart, sharing timestamps like `IMG20200728180147.jpg` or camera shutter indices like `IMG_8235.JPG`) with explicit angle/repetition markers (`_01`, `_02`).

Naive uniform random splitting would scatter consecutive shots of the same physical fruit across training and test splits. This would cause severe **data leakage**, allowing the neural network to memorize lighting and background artifacts rather than biological defect pathology.

### Multi-Criteria Group-Aware Splitting
`scripts/prepare_dataset.py` groups related frames using:
1. **Timestamp Clustering:** Timestamps within $\Delta t \le 60\text{ seconds}$ are clustered into a single atomic burst group.
2. **Camera Index Proximity:** Shutter index sequences within $\Delta \text{index} \le 3$ are clustered together.
3. **Suffix Grouping:** Base stems stripped of `_01`, `_02` are joined into the same group.

### Split Partitions (Target ~800 Images)
For the 4-class industrial inspection experiment (`Apple_Bad`, `Apple_Good`, `Banana_Bad`, `Banana_Good`):
- **Train (70%):** 592 images (67.6%)
- **Validation (15%):** 146 images (16.7%)
- **Test (15%):** 138 images (15.8%)
- **Total Prepared Samples:** 876 images

| Class Name | Train | Validation | Held-Out Test | Total |
| :--- | :---: | :---: | :---: | :---: |
| `Apple_Bad` | 141 | 37 | 43 | 221 |
| `Apple_Good` | 141 | 47 | 32 | 220 |
| `Banana_Bad` | 167 | 32 | 31 | 230 |
| `Banana_Good` | 143 | 30 | 32 | 205 |
| **Total** | **592** | **146** | **138** | **876** |

### Mechanical Invariants
- $\text{Train}_{\text{groups}} \cap \text{Val}_{\text{groups}} = \emptyset$ (0 overlap)
- $\text{Train}_{\text{groups}} \cap \text{Test}_{\text{groups}} = \emptyset$ (0 overlap)
- $\text{Val}_{\text{groups}} \cap \text{Test}_{\text{groups}} = \emptyset$ (0 overlap)
- $\text{Train}_{\text{hashes}} \cap \text{Test}_{\text{hashes}} = \emptyset$ (0 byte overlap)

Verified by automated pytest test suite in `tests/test_dataset.py`.

---

## 4. How to Reproduce Dataset Preparation

Run the preparation script from the repository root:
```bash
python scripts/prepare_dataset.py --source dataset --output dataset/processed --samples-per-class 200 --seed 42
```

Generated outputs:
- `dataset/processed/train/<class>/`
- `dataset/processed/val/<class>/`
- `dataset/processed/test/<class>/`
- `dataset/processed/manifest.json`
- `dataset/processed/split_statistics.json`
- `dataset/splits/split_manifest.json`

