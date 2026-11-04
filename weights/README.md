# FruitVision AI — Model Weights & Checkpoints Directory

This directory stores trained CNN model weights for FruitVision AI.

## Supported Formats:
1. `best.pt`: Active PyTorch EfficientNetB0 checkpoint (Auto-loaded on startup)
2. `fruit_efficientnet_b0.pt`: Backup / release checkpoint
3. `fruit_efficientnet_b0.h5` / `fruit_efficientnet_b0.keras`: Pretrained Keras/TensorFlow exports from Google Colab

## How to use trained weights from Google Colab:
1. Open `training/train.ipynb` in Google Colab (with GPU acceleration enabled).
2. Run all cells to train EfficientNetB0 on Kaggle / Fruits-360 / Defect datasets.
3. Download the resulting `best.pt` or `fruit_efficientnet_b0.pt` checkpoint.
4. Place the downloaded file in this `weights/` folder.
5. In the FruitVision web UI header, click `[RELOAD]`, or trigger the endpoint `POST /api/reload-model`. The backend will hot-reload the weights instantly without server downtime!

## Target Specifications:
- Architecture: `EfficientNetB0`
- Input Resolution: `224x224x3`
- Pretrained Base: `ImageNet-1K`
- Target Test Accuracy: `>= 90.0%`
- Classification Categories:
  - 0: `Good` (Market Grade A)
  - 1: `Damaged` (Bruised / Scratched / Impact lesion)
  - 2: `Rotten` (Decayed / Fungal spoilage)

