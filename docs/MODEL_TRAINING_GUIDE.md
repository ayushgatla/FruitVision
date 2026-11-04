# FruitVision AI — Model Training Guide

## 1. Model Architecture

FruitVision AI employs **EfficientNet-B0** pretrained on ImageNet-1K with a custom multi-layer classification head:

```text
Input (3 x 224 x 224)
   │
EfficientNet-B0 Feature Extractor (1280 output features)
   │
Adaptive AvgPool2d (1 x 1)
   │
Dropout (p=0.3)
   │
Linear (1280 -> 128)
   │
ReLU()
   │
Dropout (p=0.2)
   │
Linear (128 -> 4)  [Apple_Bad, Apple_Good, Banana_Bad, Banana_Good]
```

### Preprocessing & Augmentations
- **Input Resolution:** 224 × 224 pixels
- **Normalization:** ImageNet statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$)
- **Training Augmentations:**
  - `RandomResizedCrop(224, scale=(0.8, 1.0))`
  - `RandomHorizontalFlip(p=0.5)`
  - `RandomVerticalFlip(p=0.2)`
  - `RandomRotation(degrees=30)`
  - `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`
- **Validation / Inference Transform:**
  - `Resize((224, 224))`
  - `ToTensor()`
  - `Normalize(mean, std)`

---

## 2. Two-Stage Transfer Learning Strategy

### Stage 1: Feature Extractor Frozen, Head Warmup
- **Frozen:** `model.features.parameters()` (`requires_grad = False`)
- **Trained:** `model.classifier.parameters()`
- **Optimizer:** `AdamW(lr=1e-3, weight_decay=1e-4)`
- **Loss:** `CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)`
- **Epochs:** 4 epochs

### Stage 2: Upper Convolutional Blocks Fine-Tuning
- **Unfrozen:** `model.features[6:].parameters()` (stages 6, 7, and 8)
- **Differential Learning Rates:**
  - Backbone features: `lr=1e-5` (10x lower to preserve low-level edge/texture representations)
  - Classifier head: `lr=1e-4`
- **Scheduler:** `CosineAnnealingLR(optimizer, T_max=epochs_stage2, eta_min=1e-6)`
- **Regularization:** Early stopping with patience=4 on validation loss
- **Epochs:** 6 epochs

---

## 3. Hardware Configuration & Execution

The training pipeline automatically detects available hardware:
- **CUDA Mode (RTX 4050 6GB VRAM):** Enables `torch.amp.autocast('cuda')` with `GradScaler` for mixed-precision acceleration. Consumes < 1.2 GB VRAM at batch size 16.
- **CPU Fallback:** Runs multi-threaded on CPU if CUDA is not available.

### Run Training Command:
```bash
python -m training.train_local --batch-size 16 --epochs1 4 --epochs2 6
```

CLI Options:
- `--batch-size`: Batch size (default: 16)
- `--epochs1`: Stage 1 warmup epochs (default: 4)
- `--epochs2`: Stage 2 fine-tuning epochs (default: 6)
- `--force-cpu`: Force CPU fallback mode

---

## 4. Empirical Training Results (Run on 2026-11-04)

- **Total Execution Time:** 240.42 seconds (4.0 minutes)
- **Stage 1 (Head Warmup):**
  - Epoch 1: Train Loss 0.6568, Acc 86.66% | Val Loss 0.3353, Acc 100.00%
  - Epoch 4: Train Loss 0.2857, Acc 97.64% | Val Loss 0.2816, Acc 100.00%
- **Stage 2 (Fine-Tuning):**
  - Epoch 5 (FT 1): Train Loss 0.2750, Acc 98.82% | Val Loss 0.2808, Acc 100.00%
  - Epoch 9 (FT 5): Train Loss 0.2545, Acc 98.65% | Val Loss 0.2621, Acc 100.00% (Best Checkpoint)
- **Saved Checkpoints:**
  - `weights/best.pt`
  - `weights/fruit_efficientnet_b0.pt`

