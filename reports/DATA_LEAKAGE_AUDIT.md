# FruitVision AI — Independent Data Leakage Audit Report

**Audit Date:** 2026-11-04  
**Auditor:** Senior ML Engineer & CV Researcher (Independent Verification)  
**Target Codebase:** `FruitVision` (Dataset preparation pipeline & split manifests)  
**Dataset Roots Audited:** `dataset/` (Raw: 6,654 images) and `dataset/processed/` (Prepared: 876 images)  

---

## 1. Executive Summary

An exhaustive, independent empirical audit was conducted on the FruitVision dataset splitting pipeline (`scripts/prepare_dataset.py`), the processed partitions (`train`, `val`, `test`), and the split manifests (`dataset/splits/split_manifest.json`, `dataset/processed/manifest.json`).

| Audit Dimension | Previous Status Claim | Independently Verified Finding | Risk Level |
| :--- | :--- | :--- | :---: |
| **Exact Byte Duplicates (SHA-256)** | 0 Duplicates Reported | **VERIFIED: 0 Duplicates across all splits** | LOW |
| **Mechanical Group Key Separation** | 0 Overlapping Group Keys | **VERIFIED: Keys do not overlap, BUT key generation is mathematically flawed** | CRITICAL |
| **Temporal Proximity ($\Delta t \le 120\text{s}$)** | Unchecked in previous audit | **50 same-class pairs between Train and Test captured $\le 120\text{s}$ apart** | CRITICAL |
| **Temporal Burst Boundary Leakage** | "Zero Leakage" Claimed | **REFUTED: Train sample `16:11:55` and Test sample `16:12:00` taken 5 seconds apart** | CRITICAL |
| **Shutter Index Modulo Leakage** | "Zero Leakage" Claimed | **REFUTED: Consecutive frames (`IMG_8039` Test vs `IMG_8040` Train) separated by bucket boundary** | CRITICAL |
| **Perceptual Near-Duplicates (dHash)** | Not evaluated across splits | **11 pairs between Test and Train have Hamming Distance $\le 4$ bits (Min: 2 bits)** | HIGH |
| **Physical Fruit Identity Overlap** | "Different fruits" Assumed | **REFUTED: The exact same physical fruits appear in both Train and Test** | CRITICAL |
| **Class Balance Across Splits** | Balanced | **VERIFIED: Near-balanced (Train: 592, Val: 146, Test: 138)** | LOW |

> [!CAUTION]
> **CRITICAL VERIFIED AUDIT CONCLUSION:**  
> The previously reported **100% held-out test accuracy is statistically compromised by burst-capture data leakage**. While there are zero byte-level identical files, the grouping function in `scripts/prepare_dataset.py` contains discretization boundary flaws that scattered burst shots of identical physical fruits across the training, validation, and testing partitions.

---

## 2. Independent Dataset Verification & Class Distribution

### 2.1 Raw Dataset Inventory (`dataset/`)
- Total Scanned Images: 6,654
- Valid, non-corrupt RGB images: 6,654 (100.0%)
- Class Distribution across 6 classes:
  - `Apple_Bad`: 1,141 images (17.15%)
  - `Apple_Good`: 1,134 images (17.04%)
  - `Banana_Bad`: 1,087 images (16.34%)
  - `Banana_Good`: 1,113 images (16.73%)
  - `Lime_Bad`: 1,085 images (16.31%)
  - `Lime_Good`: 1,094 images (16.44%)
- Maximum class imbalance in raw data: 1.05:1 (Near-perfect distribution).

### 2.2 Processed Partitions (`dataset/processed/`)
The active production model was trained on a 4-class subset targeting ~200 samples per class:

| Class Name | Train Split | Validation Split | Test Split | Total Prepared | Class Share |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Apple_Bad` | 141 | 37 | 43 | 221 | 25.23% |
| `Apple_Good` | 141 | 47 | 32 | 220 | 25.11% |
| `Banana_Bad` | 167 | 32 | 31 | 230 | 26.26% |
| `Banana_Good` | 143 | 30 | 32 | 205 | 23.40% |
| **Total** | **592 (67.6%)** | **146 (16.7%)** | **138 (15.8%)** | **876** | **100.0%** |

---

## 3. Root Cause Analysis: The Group-Based Split Flaws

`scripts/prepare_dataset.py` implemented `extract_image_group_key()` to prevent data leakage from consecutive frames. However, an independent mathematical inspection reveals **two catastrophic discretization boundary flaws**:

```python
# From scripts/prepare_dataset.py (lines 42-63)
# Flaw 1: Discretization by rigid 1-minute bucket
minute_bucket = dt.strftime("%Y%m%d_%H%M")
return f"{class_name}_burst_{minute_bucket}"

# Flaw 2: Integer division of shutter index by 4
idx_bucket = idx_val // 4
return f"{class_name}_idx_{idx_bucket}"
```

### Flaw 1: Clock-Minute Truncation Boundary
- When an experimenter takes continuous photos of a single apple across minute rollovers (e.g. from 16:11:45 to 16:12:15):
  - Photos taken at `16:11:51`, `16:11:55` are assigned key: `Apple_Good_burst_20190809_1611`.
  - Photos taken at `16:12:00`, `16:12:20` are assigned key: `Apple_Good_burst_20190809_1612`.
- `prepare_dataset.py` treats these two string keys as entirely distinct independent groups.
- During `random.shuffle(groups)`, group `1611` was routed to `train` while group `1612` was routed to `test`.
- **Empirical Evidence:**
  - Train: `20190809_161155.jpg`
  - Test: `20190809_161200.jpg`
  - **Time delta: Exactly 5.0 seconds!**
  - **Empirical Count:** An independent search found **50 same-class pairs** between Train and Test captured within 120 seconds of each other.

### Flaw 2: Shutter Index Modulo-4 Truncation Boundary
- In the `Banana_Bad` and `Banana_Good` classes, images were named by camera shutter sequence numbers (e.g., `IMG_8039.JPG`, `IMG_8040.JPG`).
- `idx_val // 4` creates artificial partition walls at multiples of 4:
  - `IMG_8039` $\rightarrow 8039 // 4 = 2009 \rightarrow$ Assigned to `Banana_Bad_idx_2009` (routed to `test`).
  - `IMG_8040` $\rightarrow 8040 // 4 = 2010 \rightarrow$ Assigned to `Banana_Bad_idx_2010` (routed to `train`).
- Frames `IMG_8039` and `IMG_8040` are literally consecutive camera shots of the exact same banana taken 1–2 seconds apart!
- One frame was trained on, and the adjacent frame was evaluated in the test set.

---

## 4. Perceptual Hash (dHash) Similarity Analysis

To verify whether images separated by split boundaries depict the identical physical specimen, 64-bit difference hashes (`dHash`) and average hashes (`aHash`) were computed across all images.

### 4.1 Hamming Distance Distribution (Test vs. Train)
- **Mean Minimum Distance from Test to Train:** 14.80 bits.
- **Minimum Observed Distance:** 2 bits (96.88% perceptual identity).
- **Pairs with Hamming Distance $\le 4$ bits:** 11 distinct pairs.

### 4.2 High-Risk Perceptually Identical Test/Train Pairs

| Test Image File | Test Class | Train Image File | Train Class | Hamming Dist | Physical Identity Finding |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `IMG_8741.JPG` | `Banana_Good` | `IMG_8744.JPG` | `Banana_Good` | **2 bits** | **Identical physical banana bundle**, identical table grain, identical lighting angle |
| `IMG_8741.JPG` | `Banana_Good` | `IMG_8745.JPG` | `Banana_Good` | **2 bits** | **Identical physical banana bundle** |
| `IMG_8742.JPG` | `Banana_Good` | `IMG_8744.JPG` | `Banana_Good` | **2 bits** | **Identical physical banana bundle** |
| `IMG_8742.JPG` | `Banana_Good` | `IMG_8745.JPG` | `Banana_Good` | **2 bits** | **Identical physical banana bundle** |
| `IMG_8039.JPG` | `Banana_Bad` | `IMG_8040.JPG` | `Banana_Bad` | **4 bits** | **Identical rotten banana**, consecutive shutter frames (8039 vs 8040) |
| `IMG_9230.JPG` | `Banana_Good` | `IMG_9236.JPG` | `Banana_Good` | **4 bits** | Same specimen, minor rotation on background plate |
| `IMG_9230.JPG` | `Banana_Good` | `IMG_9237.JPG` | `Banana_Good` | **4 bits** | Same specimen, minor rotation |
| `IMG_9230.JPG` | `Banana_Good` | `IMG_9238.JPG` | `Banana_Good` | **4 bits** | Same specimen, minor rotation |
| `IMG_9231.JPG` | `Banana_Good` | `IMG_9236.JPG` | `Banana_Good` | **4 bits** | Same specimen, minor rotation |

---

## 5. Audit Categorization & Decision Factors

### Verified Results
1. Exact byte duplicate count is **zero (0)** across all raw and processed partitions.
2. The mechanical assertion in `scripts/prepare_dataset.py` passed only because it tested disjointness of generated string keys (`assigned_groups['train'] & assigned_groups['test'] == set()`), not physical fruit independence.
3. Class balance across splits is healthy (each class constitutes 23% to 26% of each partition).

### Unverified / Refuted Claims
1. **Refuted Claim:** *"Group-aware splitting eliminated data leakage from burst captures."*  
   **Reality:** Clock-minute truncation and modulo-4 integer division leaked at least 50 near-temporal shots and 11 perceptual near-duplicates into the test set.
2. **Refuted Claim:** *"100% test accuracy proves human-level generalized defect discrimination."*  
   **Reality:** High test accuracy is heavily aided by memorization of identical fruit backgrounds, specular table reflections, and specimen geometries seen during training.

### Known Limitations
1. The raw agricultural dataset lacks explicit metadata files (no CSV of fruit specimen IDs).
2. Clustered shooting sessions can only be identified by continuous temporal proximity or continuous shutter sequence numbering.

### Recommended Corrections
1. **Continuous Rolling-Window Grouping:**  
   Replace discrete minute binning (`strftime('%Y%m%d_%H%M')`) with continuous hierarchical distance clustering:
   $$\Delta t \le 300\text{ seconds (5 minutes)} \implies \text{Merge into same cluster}$$
2. **Shutter Sequence Clearance:**  
   Replace `idx // 4` with connected component grouping: any camera indices with $|idx_a - idx_b| \le 10$ must belong to the same cluster.
3. **Cluster-Level Stratification:**  
   Split entire clusters, not individual buckets, ensuring a minimum time gap of at least 15 minutes between any training specimen and any test specimen.

