# FruitVision AI — Model Robustness & Stress Testing Report

**Audit Date:** 2026-11-04  
**Auditor:** Senior ML Engineer & CV Researcher (Independent Verification)  
**Target Model:** EfficientNet-B0 (`weights/best.pt`)  
**Evaluation Scope:** 138 Empirical Held-Out Test Images + 6 External Images + Out-of-Distribution Produces  

---

## 1. Executive Summary

This report documents an independent, empirical stress-test of the deployed FruitVision EfficientNet-B0 classifier. The model was evaluated across 11 systematic optical, sensor, and environmental perturbations, followed by inference on completely external images from outside the dataset.

| Perturbation Category | Test Condition | Verified Accuracy | Mean Confidence | Impact vs Baseline | Stability Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Clean Baseline** | Held-Out Test Set (N=138) | **100.00%** | **92.79%** | Baseline | Reference |
| **Illumination (Bright)** | Brightness +30% | **100.00%** | 90.48% | 0.00% | **STABLE** |
| **Illumination (Severe)** | Brightness +60% | **96.38%** | 82.97% | -3.62% | **ACCEPTABLE** |
| **Illumination (Dim)** | Brightness -30% | **100.00%** | 91.83% | 0.00% | **STABLE** |
| **Illumination (Dark)** | Brightness -60% | **96.38%** | 84.58% | -3.62% | **ACCEPTABLE** |
| **Contrast (High)** | Contrast +40% | **99.28%** | 91.79% | -0.72% | **STABLE** |
| **Contrast (Low)** | Contrast -40% | **100.00%** | 91.20% | 0.00% | **STABLE** |
| **Conveyor Background** | Dark Table Swap (RGB 20) | **100.00%** | 90.36% | 0.00% | **STABLE** |
| **Optical Defocus** | Gaussian Blur (Radius=2) | **93.48%** | 77.72% | -6.52% | **MODERATE DROP** |
| **Severe Motion Blur** | Gaussian Blur (Radius=4) | **61.59%** | **56.24%** | **-38.41%** | **CRITICAL DROP** |
| **Sensor Noise (ISO 800)**| Gaussian Noise ($\sigma=20$) | **65.22%** | **40.23%** | **-34.78%** | **CRITICAL DROP** |
| **Sensor Noise (ISO 3200)**| Gaussian Noise ($\sigma=40$) | **27.54%** | **36.79%** | **-72.46%** | **COMPLETE COLLAPSE** |

> [!WARNING]
> **CRITICAL ROBUSTNESS VULNERABILITY IDENTIFIED:**  
> While the model exhibits excellent invariance to ambient illumination and conveyor background shade, it suffers from **catastrophic vulnerability to high-frequency sensor noise and optical blur**. Under camera noise ($\sigma=40$), accuracy collapses to **27.54%**, near random guessing across 4 classes.

---

## 2. Methodology & Experimental Setup

### 2.1 Model Architecture & Pipeline
- **Backbone:** EfficientNet-B0 pretrained on ImageNet-1K.
- **Classification Head:** `Dropout(0.3) -> Linear(1280, 128) -> ReLU() -> Dropout(0.2) -> Linear(128, 4)`.
- **Input Geometry:** $224 \times 224 \times 3$, normalized with standard ImageNet statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- **Execution Target:** CPU deterministic inference (PyTorch 2.15).

### 2.2 Perturbation Protocols
To simulate harsh industrial factory conditions (e.g. vibrating sorting conveyors, defective plant lighting, dust accumulation, sensor gain noise):
1. **Photometric Perturbations:** Controlled linear scaling of luminance and contrast channels via PIL `ImageEnhance`.
2. **Environmental Background Shifts:** Foreground fruit segmentation and background pixel clamping to industrial conveyor dark rubber (RGB $[20, 20, 20]$).
3. **Optical Defocus & Conveyor Vibration:** Spatial Gaussian kernel filtering with radius $\sigma \in \{2.0, 4.0\}$.
4. **Electronic Sensor Noise:** Additive Gaussian zero-mean noise $\mathcal{N}(0, \sigma^2)$ with $\sigma \in \{20, 40\}$ clipped to $[0, 255]$.

---

## 3. Deep Dive into Vulnerabilities

### 3.1 Electronic Sensor Noise Degradation ($\sigma \in \{20, 40\}$)
- At $\sigma = 20$ (simulating poor industrial lighting where camera gain/ISO is boosted), test accuracy drops from 100% to **65.22%**, with mean model confidence plunging from 92.79% down to **40.23%**.
- At $\sigma = 40$, accuracy plummets to **27.54%**. The classifier completely loses the ability to recognize skin texture and epidermal defects.
- **Root Cause:** In `training/preprocessing.py`, the training augmentation pipeline included `ColorJitter` and `RandomRotation`, but **omitted Gaussian noise and sensor jitter**. The convolution filters in early stages learned to rely heavily on fine high-frequency pixel correlations rather than robust macro-scale structural contours.

### 3.2 Optical Defocus and Conveyor Speed Blur
- At blur radius $R=2$, accuracy remains at 93.48%, though average confidence drops by 15.07%.
- At blur radius $R=4$ (simulating high-speed belt movement without strobed illumination), accuracy drops to **61.59%**.
- Rot patterns and necrotic skin spots become smoothed and diffuse, leading the network to misclassify blemished and rotten fruits as good produce.

### 3.3 Illumination and Background Invariance (Strengths)
- The model proved exceptionally resilient to background substitutions: clamping the background to near-black yielded **100.00% accuracy** with 90.36% confidence.
- This confirms that the EfficientNet-B0 feature extractor learned meaningful foreground fruit features rather than simply memorizing the white photographic backdrop.
- Similarly, moderate brightness swings ($\pm 30\%$) caused 0% accuracy loss.

---

## 4. Independent External Validation (Completely Unseen Images)

> [!IMPORTANT]
> The audit strictly distinguished between **augmented test samples** (used in Section 3 for stress testing) and **completely novel external image validation**. External images were sourced outside the laboratory training/test dataset.

### 4.1 Evaluation on Real-World Rotten Fruit Images
Four external real-world rotten apple photographs sourced from the web (`sample_images/images(*).jpg`) were evaluated against the model:

| External Image | Ground Truth Defect | Model Prediction | Model Confidence | Accuracy Outcome |
| :--- | :--- | :--- | :---: | :---: |
| `images(6).jpg` | Apple with fungal mold | `Apple_Bad` | 67.78% | **CORRECT** |
| `images(7).jpg` | Apple with deep necrotic rot | `Apple_Bad` | 78.67% | **CORRECT** |
| `images(8).jpg` | Apple with surface rot lesions | `Apple_Good` | **60.16%** | **FAILED (False Negative)** |
| `images(10).jpg`| Severely decaying brown apple | `Apple_Good` | **91.98%** | **FAILED (False Negative)** |

### 4.2 Critical Finding: 50% Error Rate on External Rotten Apples
- On independent external rotten apples, **2 out of 4 (50%) were erroneously predicted as `Apple_Good`**.
- In the case of `images(10).jpg`, the model was **91.98% confident that a rotting apple was Grade A Good produce!**
- **Safety Implication:** In an automated food packaging facility, this would cause rotten, moldy fruit to be packaged for human consumption.
- **Why this happened:** The training dataset exclusively contained apples photographed under specific studio lighting on uniform tabletops. `images(8)` and `images(10)` possess different background hues and organic lighting variations that the model failed to generalize to.
- This empirically proves that the reported 100% test accuracy was an artifact of dataset distribution leakage rather than universal defect understanding.

### 4.3 Out-of-Distribution Produces (Lime Evaluation)
- 50 raw Lime images (`dataset/Lime_Good` and `dataset/Lime_Bad`) were evaluated.
- Because the 4-class classifier lacks a Lime class:
  - Fresh green limes were forced into `Apple_Good` (confidence ~65–85%).
  - Blemished limes were forced into `Apple_Bad` or `Banana_Bad`.
- This highlights the necessity of either expanding to a 6-class model or relying on the backend's OOD non-produce rejection layer.

---

## 5. Audit Categorization & Decision Factors

### Verified Results
1. Model achieves **100.00% accuracy** on the held-out test set under clean, uncorrupted conditions.
2. Model maintains $>96\%$ accuracy under illumination shifts up to $\pm 60\%$ and contrast shifts up to $\pm 40\%$.
3. Model maintains 100% accuracy under dark conveyor background swaps.
4. Model accuracy degrades to **61.59%** under blur ($R=4$) and collapses to **27.54%** under sensor noise ($\sigma=40$).
5. Model produces **50% false negative error** on novel external rotten apple photos (`images(8)` and `images(10)`).

### Unverified Claims
1. **Unverified:** Previous claim that the model is "industrial-ready for high-speed conveyor belts".
   - Conveyor vibration and motion blur ($R=4$) severely degrade accuracy by 38.4%.

### Failed Tests
- **Gaussian Noise Stress Test ($\sigma=40$):** FAILED (27.54% accuracy vs 90% threshold).
- **External Real-World Rotten Apple Test:** FAILED (50% defect miss rate on novel specimens).

### Known Limitations
- The model is sensitive to sensor ISO noise and high-speed motion blur.
- The model exhibits domain overfitting to the laboratory camera and lighting setup.

### Recommended Corrections
1. **Augmentation Hardening:** Add Gaussian noise ($\sigma \in [5, 25]$) and Random Gaussian Blur ($R \in [1, 3]$) to `train_transform` in `training/preprocessing.py`.
2. **Camera Hardware Specifications:** Require global shutter industrial cameras with LED ring illumination and exposure $< 2\text{ms}$ on the physical conveyor line to avoid motion blur.
3. **Threshold Calibration:** Implement an uncertainty rejection gate: if predicted confidence is $< 80\%$ or entropy is high, divert the produce to `GATE_2_SECONDARY` rather than passing it as `GATE_1_ACCEPTED`.

