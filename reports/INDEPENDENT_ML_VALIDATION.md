# FruitVision AI — Independent Machine Learning Validation Report

**Audit Date:** 2026-11-04  
**Auditor:** Senior Machine Learning Engineer & Computer Vision Researcher  
**Target Checkpoint:** `weights/best.pt` (SHA-256: `6aa4705bc8ac7ee9caca8e87fd3e5ab8b7fb3d7c528ff0878b783ac1f31d719e`)  
**Secondary Checkpoint:** `weights/fruit_efficientnet_b0.pt` (SHA-256: `a3a551b1168c61b0bfb027cb0c7fcc56f642828ef301deeb24345fdf6c80661b`)  
**Architecture:** EfficientNet-B0 (Transfer Learning with Custom Classification Head)  
**Evaluation Scope:** 138 Empirical Held-Out Test Samples, Calibration, Preprocessing Audit, Label Semantics, External Unseen Testing  

---

## 1. Executive Summary

This audit independently validates the Machine Learning pipeline, model checkpoint, classification performance, and statistical reliability of FruitVision AI.

| Evaluation Metric / Check | Reported Value | Independently Reproduced Value | Verification Status |
| :--- | :---: | :---: | :---: |
| **Held-Out Test Accuracy** | 100.00% | **100.00% (138 / 138)** | **REPRODUCED** |
| **Macro Precision** | 100.00% | **100.00%** | **REPRODUCED** |
| **Macro Recall** | 100.00% | **100.00%** | **REPRODUCED** |
| **Macro F1-Score** | 100.00% | **100.00%** | **REPRODUCED** |
| **Weighted F1-Score** | 100.00% | **100.00%** | **REPRODUCED** |
| **Expected Calibration Error (ECE)** | Not Reported | **7.21%** | **MEASURED** |
| **Brier Score** | Not Reported | **0.0054** | **MEASURED** |
| **Mean Max Confidence** | Not Reported | **92.79%** | **MEASURED** |
| **Defect Miss Rate (FNR on Bad)** | 0.00% | **0.00% (on test set)** | **REPRODUCED** |
| **External Rotten Apple Accuracy** | Not Reported | **50.00% (2/4 misclassified)** | **CRITICAL LIMITATION** |
| **Preprocessing Parity** | ImageNet Norm | **Verified Exact Match** | **PASSED** |
| **Label Semantics Integrity** | Binary Quality | **Bad Quality Preserved** | **PASSED** |

> [!IMPORTANT]
> **KEY EMPIRICAL TAKEAWAY:**  
> The 100% test accuracy reported by the previous engineering run **is strictly reproducible on the saved checkpoint (`weights/best.pt`) using the test split**. However, this result reflects **memorization of burst-capture specimens** that leaked across minute/index boundary splits (as documented in `reports/DATA_LEAKAGE_AUDIT.md`). When evaluated against completely external real-world rotten fruit images from outside the dataset, the model's true accuracy drops to **50.00%**.

---

## 2. Checkpoint & Architecture Inspection

### 2.1 Checkpoint Integrity (`weights/best.pt`)
- **File Size:** 16,948,443 bytes (~16.9 MB)
- **Top-Level Keys:**
  - `model_state_dict`: Complete tensor dictionary for backbone and head.
  - `classes`: `['Apple_Bad', 'Apple_Good', 'Banana_Bad', 'Banana_Good']`
  - `class_to_idx`: `{'Apple_Bad': 0, 'Apple_Good': 1, 'Banana_Bad': 2, 'Banana_Good': 3}`
  - `val_accuracy`: `1.0` (Best recorded epoch: 9)
  - `val_loss`: `0.0004`
  - `architecture`: `EfficientNetB0`
  - `model_version`: `2.0.0-efficientnet-b0-4cls`
  - `input_resolution`: `[224, 224, 3]`
  - `normalization`: `{'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}`
  - `device_used`: `cpu`

### 2.2 Neural Network Architecture
The classification head replaces the default ImageNet 1000-class linear layer:
```text
EfficientNet-B0 Backbone (Features: 1280 output dimensions)
  │
  ├── Dropout(p=0.3, inplace=True)
  ├── Linear(in_features=1280, out_features=128, bias=True)
  ├── ReLU()
  ├── Dropout(p=0.2)
  └── Linear(in_features=128, out_features=4, bias=True)
```
- Total trainable parameters in head: $1280 \times 128 + 128 + 128 \times 4 + 4 = 164,484$ parameters.
- Architecture matches throughout training (`build_model`) and serving (`FruitVisionDetector`).

---

## 3. Independent Reproduction on Held-Out Test Set

The model was loaded into an isolated CPU session and evaluated on all 138 test images in `dataset/processed/test/`.

### 3.1 Per-Class Breakdown

| Class Name | Support (N) | Precision | Recall | F1-Score | True Positive | False Positive | False Negative |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Apple_Bad` | 43 | 1.0000 | 1.0000 | 1.0000 | 43 | 0 | 0 |
| `Apple_Good` | 32 | 1.0000 | 1.0000 | 1.0000 | 32 | 0 | 0 |
| `Banana_Bad` | 31 | 1.0000 | 1.0000 | 1.0000 | 31 | 0 | 0 |
| `Banana_Good` | 32 | 1.0000 | 1.0000 | 1.0000 | 32 | 0 | 0 |
| **Macro Average** | **138** | **1.0000** | **1.0000** | **1.0000** | **138** | **0** | **0** |
| **Weighted Average**| **138** | **1.0000** | **1.0000** | **1.0000** | **138** | **0** | **0** |

### 3.2 Confusion Matrix
$$\begin{pmatrix}
43 & 0 & 0 & 0 \\
0 & 32 & 0 & 0 \\
0 & 0 & 31 & 0 \\
0 & 0 & 0 & 32
\end{pmatrix}$$
- 0 off-diagonal misclassifications.
- False Negative Rate on Defective / Bad Produce: **0.00%** on this partition.

### 3.3 Confidence Calibration & Uncertainty
- **Expected Calibration Error (ECE):** 7.21% across 10 confidence bins.
- **Brier Score Loss:** 0.0054 (near optimal for confident binary separations).
- **Mean Predicted Confidence:** 92.79%.
- The model outputs steep softmax distributions (>90% confidence on almost all in-distribution test samples).

---

## 4. Preprocessing & Serving Alignment Audit

A common failure mode in Computer Vision deployments is training-serving skew (color channels, resize algorithms, or normalization mismatches).

| Preprocessing Step | Training Pipeline (`training/`) | Serving Detector (`backend/services/detector.py`) | Alignment Verdict |
| :--- | :--- | :--- | :---: |
| **Color Space Conversion** | OpenCV BGR $\rightarrow$ PIL RGB via `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` | OpenCV BGR $\rightarrow$ PIL RGB via `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` | **EXACT MATCH** |
| **Resize Geometry** | $224 \times 224$ (Bilinear interpolation) | $224 \times 224$ (`transforms.Resize((224, 224))`) | **EXACT MATCH** |
| **Tensor Scaling** | $[0, 255] \rightarrow [0.0, 1.0]$ via `transforms.ToTensor()` | $[0, 255] \rightarrow [0.0, 1.0]$ via `transforms.ToTensor()` | **EXACT MATCH** |
| **Normalization Mean** | $[0.485, 0.456, 0.406]$ | $[0.485, 0.456, 0.406]$ | **EXACT MATCH** |
| **Normalization Std** | $[0.229, 0.224, 0.225]$ | $[0.229, 0.224, 0.225]$ | **EXACT MATCH** |
| **Batch Dimension** | `unsqueeze(0)` on inference crop | `unsqueeze(0)` on inference crop | **EXACT MATCH** |

---

## 5. Dataset Label Semantics Audit (Phase 3)

### 5.1 Dataset Origin and Ground Truth Nomenclature
- The raw dataset consists of folders: `Apple_Bad`, `Apple_Good`, `Banana_Bad`, `Banana_Good`, `Lime_Bad`, `Lime_Good`.
- There is **no ground-truth 3-class "Good vs Damaged vs Rotten" annotation** in the dataset.
- Inspection of `Apple_Bad` and `Banana_Bad` images reveals:
  - Some specimens show dark fungal decay (botrytis/mold).
  - Other specimens exhibit mechanical bruises, cuts, peel abrasions, or severe brown skin freckling (overripeness).
- Therefore, the term **"Bad" represents a broad industrial quality rejection category** encompassing both damage and rot.

### 5.2 Application Semantic Handling
- **Preservation of Bad Quality:** The system does **not invent a fake third neural network class** (which would corrupt ground truth).
- **Dual-Tier Mapping:**
  - Raw prediction: Preserves fine-grained class (`Apple_Bad`, `Banana_Bad`).
  - Quality mapping: Categorizes `Bad` as `REJECTED` / `Bad Quality`.
  - Heuristic safety layer: If optical surface analysis detects low-grade superficial bruising ($8\% \le \text{rot} < 25\%$), the application flags it as `Damaged` / `SECONDARY_INSPECTION` for manual trimming.
- This preserves the required food safety conservatism without training a model on imaginary labels.

---

## 6. External Real-World Generalization (Out-of-Dataset Validation)

When evaluated on novel external images not sourced from the dataset:
- `images(6).jpg` (Rotten Apple): **Apple_Bad** (67.78% conf) — Correct
- `images(7).jpg` (Rotten Apple): **Apple_Bad** (78.67% conf) — Correct
- `images(8).jpg` (Rotten Apple): **Apple_Good** (60.16% conf) — **FAILED**
- `images(10).jpg` (Rotten Apple): **Apple_Good** (91.98% conf) — **FAILED**

### Why External Generalization Drops:
1. **Lighting & Specular Reflection:** The laboratory dataset used diffuse top lighting on a white tabletop. `images(10)` has natural ambient daylight, causing the model to misinterpret the dark rot patch as a shadow and the lighter peel as fresh produce.
2. **Peel Contrast:** When background contrast deviates from the training studio, the network struggles to separate background from foreground without an explicit segmentation mask.

---

## 7. Audit Categorization & Decision Factors

### Verified Results
1. Saved checkpoint `weights/best.pt` reliably produces 100% accuracy on `dataset/processed/test/`.
2. Checkpoint weights and inference transforms are fully aligned between training and FastAPI runtime.
3. Class ordering `[Apple_Bad, Apple_Good, Banana_Bad, Banana_Good]` is strictly consistent across training, checkpoint metadata, and API routes.
4. ECE is 7.21% and Brier score is 0.0054.

### Unverified Claims
1. **Claim:** *"The model achieves 100% generalized accuracy in real-world produce sorting."*  
   **Reality:** On novel web images of rotten produce, the model had a **50% defect miss rate**, misclassifying rotten apples as Grade A Good.

### Failed Tests
- Novel external rotten apple generalization: 2/4 failed.

### Known Limitations
- The 4-class model cannot identify or classify other fruits (e.g. Limes, Oranges) without forcing them into Apple or Banana.
- Over-reliance on background and lighting patterns seen during training.

### Recommended Corrections
1. Retrain on a larger, multi-source dataset combining Fruits-360 with open-source agricultural defect datasets (e.g. Kaggle Rotten Fruit Dataset).
2. Augment training with random color balance, background swap, and synthetic shadow injection.
3. Enforce temperature scaling on softmax logits to bring calibration error (ECE) below 3.0%.

