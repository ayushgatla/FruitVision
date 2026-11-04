# FruitVision AI — Comprehensive Machine Learning Engineering (MLE) Audit

**Audit Date:** 2026-11-04  
**Auditor:** Senior Machine Learning Engineer Reviewer (mle-reviewer)  
**Repository:** FruitVision (c:\Users\ramsa\Desktop\FruitVision)  
**Verdict:** **BLOCK (Production Promotion Blocked)**  
**Primary Risks:** Burst-capture train/test leakage | Memorization masked by 100% test accuracy | Critical defect miss rate (50%) on real-world novel rotten fruit | Sensor noise and blur instability | Closed-world OOD vulnerability.

---

## 1. Executive Summary & Review Verdict

An in-depth, rigorous Machine Learning validation audit was performed on the FruitVision repository, examining dataset pipelines, model checkpoints, statistical reliability, inference behavior, and deployment robustness.

`	ext
[CRITICAL] Severe Burst-Capture Data Leakage Between Train and Test Splits
File: scripts/prepare_dataset.py:42-64
Issue: Discrete 1-minute time bucketing and modulo-4 integer division split burst captures of the exact same physical fruit specimens across train, validation, and test splits. 51.45% of the test set is contaminated with train leakage.
Fix: Replace discrete string bucketing with continuous rolling-window hierarchical clustering (merge if delta_t <= 300s or camera index delta <= 10 or dHash <= 6).

[HIGH] 50% Defect Miss Rate on Real-World Novel Rotten Produce
File: weights/best.pt & backend/services/detector.py:280-360
Issue: Model achieves 100% on leaked test set, but misclassifies 2 of 4 external real-world rotten apples as Grade A Good (with up to 91.98% confidence).
Fix: Retrain on the full 6,654 images; implement temperature scaling and an out-of-distribution confidence rejection gate.

[HIGH] Catastrophic Vulnerability to Sensor Noise and High-Speed Motion Blur
File: training/preprocessing.py:24-33
Issue: Accuracy drops to 61.59% under blur (radius=4) and collapses to 27.54% under sensor noise (std=40). Augmentation omitted high-frequency noise and blur.
Fix: Add Gaussian noise (std in [5, 25]) and Gaussian blur (radius in [1, 3]) to train_transform.

[MEDIUM] Unutilized Dataset Potential (86.8% Discarded)
File: scripts/prepare_dataset.py:27-29
Issue: 5,778 raw images (86.8% of available data) and all Lime classes are discarded to form an arbitrary 4-class ~200-sample dataset.
Fix: Expand classification to all 6 classes (Apple, Banana, Lime) or adopt multi-task fruit-type + defect grading.
`

---

## 2. Dataset Integrity & Data Leakage Assessment

### 2.1 Raw Dataset vs. Processed Subset
- **Raw Repository Assets:** 6,654 images across 6 classes: Apple_Bad (1,141), Apple_Good (1,134), Banana_Bad (1,087), Banana_Good (1,113), Lime_Bad (1,085), Lime_Good (1,094).
- **Exact Byte Duplicates:** 0 SHA-256 collisions across all 6,654 images.
- **Prepared Subset:** 876 images partitioned into Train (592), Val (146), Test (138).
- **Data Gap:** 5,778 raw images (86.8%) were excluded, including 2,179 Lime images.

### 2.2 Mathematical Flaws in Group-Aware Splitting
scripts/prepare_dataset.py implemented extract_image_group_key() to prevent data leakage, but two discrete bucketing flaws caused cross-split contamination:

1. **Clock-Minute Discretization Flaw (dt.strftime('%Y%m%d_%H%M')):**
   - Burst captures spanning minute rollovers (e.g., 16:11:55 in Train and 16:12:00 in Test) were assigned different string group keys.
   - **Empirical Evidence:** 50 same-class pairs between Train and Test were captured within 120 seconds of each other.
2. **Camera Sequence Modulo-4 Flaw (idx // 4):**
   - Sequential frames crossing multiples of 4 (e.g. IMG_8039 in Test and IMG_8040 in Train) were assigned to different buckets.
   - **Perceptual Similarity:** dHash Hamming distance as low as 2 bits (e.g. IMG_8741 vs IMG_8744, Pearson correlation 0.980, MAE 8.14 intensity levels).
3. **Contamination Extent:** 71 of 138 test samples (51.45%) have direct burst or perceptual leakage from the training set.

---

## 3. Model Validation, Calibration & Generalization vs Memorization

### 3.1 Test Set Reproduction (weights/best.pt)
- **Test Accuracy:** 100.00% (138 / 138)
- **Macro Precision / Recall / F1:** 1.0000 / 1.0000 / 1.0000
- **Brier Score:** 0.0054
- **Expected Calibration Error (ECE):** 7.21%
- **Mean Max Confidence:** 92.79%

### 3.2 Memorization Diagnosis
- Leaked test subset (N=71) mean confidence: **94.34%**.
- Clean test subset (N=67) mean confidence: **91.15%**.
- External real-world evaluation on web images (sample_images/):
  - images(6).jpg: Correct (Apple_Bad, 67.78%)
  - images(7).jpg: Correct (Apple_Bad, 78.67%)
  - images(8).jpg: **FAILED (Apple_Good, 60.16%)**
  - images(10).jpg: **FAILED (Apple_Good, 91.98%)**
- **True Defect Miss Rate on External Rotten Fruit: 50.00%**.

---

## 4. Robustness Profile Under Image Degradation

| Stress Condition | Parameter | Accuracy | Confidence | Degradation vs Baseline | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Clean Baseline** | Held-Out Test Set | 100.00% | 92.79% | Baseline | Reference |
| **Illumination** | Brightness +-30% | 100.00% | ~91.0% | 0.00% | Robust |
| **Severe Illumination**| Brightness +-60% | 96.38% | ~83.5% | -3.62% | Acceptable |
| **Contrast** | Contrast +-40% | 99.28%-100% | ~91.5% | -0.72% | Robust |
| **Conveyor Background**| Dark Rubber Swap | 100.00% | 90.36% | 0.00% | Robust |
| **Optical Blur** | Gaussian Blur R=2 | 93.48% | 77.72% | -6.52% | Moderate Drop |
| **Conveyor Vibration** | Gaussian Blur R=4 | 61.59% | 56.24% | -38.41% | Critical Drop |
| **Sensor Noise** | Gaussian Noise std=20 | 65.22% | 40.23% | -34.78% | Critical Drop |
| **Low-Light Noise** | Gaussian Noise std=40 | 27.54% | 36.79% | -72.46% | Complete Collapse |

---

## 5. Dataset Label Semantics & Closed-World Risks

- **Ground Truth Semantics:** Dataset contains only binary quality labels ([Fruit]_[Good] vs [Fruit]_[Bad]). "Bad" encompasses fungal rot, cuts, bruises, and skin abrasions.
- **Serving Heuristic Skew:** detector.py contains a heuristic HSV color segmentation rule that overrides model predictions if rot area > 8%. This heuristic masks model false positives in the API, but is absent during model training.
- **Closed-World Failure:** When 50 raw Lime images were fed into the 4-class classifier, fresh green limes were forced into Apple_Good with up to 97% confidence.

---

## 6. Actionable Recommendations & Remediation Roadmap

1. **Continuous Rolling-Window Grouping:** Replace discrete minute and modulo-4 binning with single-linkage distance clustering (merge if delta_t <= 300s, index delta <= 10, or dHash <= 6), with a mandatory 15-minute split buffer.
2. **Train on Full 6,654 Dataset:** Expand model to 6 classes (incorporating Lime_Good and Lime_Bad) or multi-task produce/defect classification.
3. **Augmentation Hardening:** Add Gaussian noise (std=5-25) and Gaussian blur (radius=1-3) to 	rain_transform.
4. **Logit Calibration & OOD Gating:** Apply temperature scaling to logits and add an uncertainty threshold (if max probability < 0.80, route to secondary manual inspection).
5. **CI Regression Tests:** Add dHash cross-split assertions and perturbation robustness gates to automated test suites.
