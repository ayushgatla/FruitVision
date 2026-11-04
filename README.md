# FruitVision AI — Intelligent Fruit Quality Inspection & Sorting System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI%200.162-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch%20%7C%20EfficientNetB0-EE4C2C.svg)](https://pytorch.org/)
[![React 19](https://img.shields.io/badge/Frontend-React%2019%20%2B%20Tailwind%20v4-61DAFB.svg)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An Industry 4.0 automated fruit quality inspection, defect analysis, and conveyor sorting simulation system powered by **Transfer-Learned EfficientNetB0**, OpenCV computer vision segmentation, and a strict monochrome industrial control room UI.

---

## System Architecture & Features

- **Transfer-Learned EfficientNetB0**: Pretrained on ImageNet-1K and fine-tuned using a 2-stage transfer learning pipeline on empirical fruit quality data (`Apple_Good`, `Apple_Bad`, `Banana_Good`, `Banana_Bad`).
- **Empirical Dataset Pipeline**: 6,654 real organic fruit images with group-aware stratified splitting that eliminates data leakage from consecutive burst frames and augmentations.
- **Fruit-Specific Conveyor Lines**: Real-time simulated sorting conveyor supporting dedicated **Apple Inspection Line**, **Banana Inspection Line**, and Full Production Mix with 3-way pneumatic divert gates (`GATE_1_ACCEPTED`, `GATE_2_SECONDARY`, `GATE_3_REJECTED`).
- **Out-of-Distribution "Not Fruits" Detector**: Multi-layer optical rejection engine that identifies non-produce artifacts (electronics, documents, vehicles, blanks) and flags them as `[NOT FRUITS]`.
- **Dual-Tier Quality Mapping**: Maps 4-class fine-grained produce quality predictions to canonical industrial grades (`ACCEPTED`, `SECONDARY_INSPECTION`, `REJECTED`) without breaking downstream schemas.
- **Multi-Fruit Batch Segmentation**: OpenCV contour detection and individual fruit isolation for conveyor batch processing.
- **Real-Time Webcam Monitor**: Live video feed processing with HUD overlay and automatic defect logging.
- **Persistent SQLite Audit Logging**: Complete historical traceability with indexed search, status filters, CSV export, and annotated visual evidence snapshots.
- **Empirical Telemetry & Quality Analytics**: Real held-out confusion matrix, per-class precision/recall, 7-day acceptance trajectories, and latency benchmarking.

---

## Documentation Guides

- [Dataset Guide (`docs/DATASET_GUIDE.md`)](docs/DATASET_GUIDE.md): Full dataset audit, class counts, image geometries, and group-aware leakage prevention.
- [Model Training Guide (`docs/MODEL_TRAINING_GUIDE.md`)](docs/MODEL_TRAINING_GUIDE.md): EfficientNet-B0 architecture, 2-stage transfer learning, hyperparameter specs, and training results.
- [Model Evaluation Guide (`docs/MODEL_EVALUATION_GUIDE.md`)](docs/MODEL_EVALUATION_GUIDE.md): Held-out test performance, per-class metrics, confusion matrix, and latency benchmarks.
- [ML Implementation Audit (`reports/ML_IMPLEMENTATION_AUDIT.md`)](reports/ML_IMPLEMENTATION_AUDIT.md): Comprehensive verification against task acceptance criteria.

---

## Commands to Run the Project

### Option A: One-Click Quick Start (Windows)
```cmd
start.bat
```
Starts:
- **FastAPI Backend Server** on `http://127.0.0.1:8001`
- **Vite React Frontend** on `http://localhost:5174`

---

### Option B: Manual Setup & Execution

#### 1. Backend Setup & Startup
```bash
# Activate virtual environment
venv\Scripts\activate

# Start FastAPI backend server
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8001 --reload
```
> Interactive OpenAPI Swagger Docs: [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

#### 2. Frontend Setup & Startup
```bash
cd frontend
npm run dev
```
> Web Application: [http://localhost:5174](http://localhost:5174)

---

## Dataset Preparation, Training & Evaluation

### 1. Group-Aware Dataset Preparation
Prepares ~800 images across 4 classes with strict 0% data leakage:
```bash
python scripts/prepare_dataset.py --source dataset --output dataset/processed --samples-per-class 200 --seed 42
```

### 2. Two-Stage EfficientNet-B0 Training
Trains classification head (Stage 1) and fine-tunes upper convolutional blocks (Stage 2):
```bash
python -m training.train_local --batch-size 16 --epochs1 4 --epochs2 6
```
Outputs saved to `weights/best.pt` and `weights/fruit_efficientnet_b0.pt`.

### 3. Held-Out Test Evaluation & Benchmarking
Evaluates exclusively on held-out test split (138 real images) and measures inference latency:
```bash
python -m training.evaluate --weights weights/best.pt --dataset dataset/processed --iterations 100
```
Generated reports:
- `reports/classification_report.json`
- `reports/benchmark_report.json`
- `reports/confusion_matrix.png`
- `reports/training_curves.png`

---

## Service Port Reference

| Service | Port | URL | Description |
| :--- | :---: | :--- | :--- |
| **Frontend UI** | `5174` | `http://localhost:5174` | Vite React Industrial Control Room |
| **Backend API** | `8001` | `http://127.0.0.1:8001` | FastAPI Uvicorn REST & Telemetry Server |
| **API Documentation** | `8001` | `http://127.0.0.1:8001/docs` | Interactive Swagger / OpenAPI Specification |

---

## Directory Structure

```text
FruitVision/
├── backend/
│   ├── api/                    # REST routes (inspect, simulation, history, analytics)
│   ├── database/               # SQLite engine, models & session handlers
│   ├── models/                 # Pydantic request/response validation schemas
│   ├── services/               # EfficientNetB0 detector & analytics service
│   ├── static/                 # Uploads, evidence snapshots & sample assets
│   ├── utils/                  # HUD drawing helpers & file handlers
│   ├── main.py                 # FastAPI application entry point
│   └── requirements.txt        # Python backend dependencies
├── dataset/                    # Real fruit image dataset (6,654 images)
│   └── processed/              # Group-stratified train/val/test splits
├── docs/                       # Comprehensive guides (dataset, training, evaluation)
├── frontend/                   # Vite React industrial UI
├── reports/                    # Empirical audit, classification, benchmark reports & plots
├── scripts/                    # Dataset audit and preparation automation
├── training/
│   ├── config.py               # Central ML configuration
│   ├── dataset.py              # PyTorch FruitDataset & DataLoaders
│   ├── train_local.py          # 2-stage transfer learning pipeline
│   ├── evaluate.py             # Empirical held-out test evaluation
│   └── preprocessing.py        # ImageNet transforms & augmentations
├── weights/                    # Active model checkpoint (best.pt) & backups
└── start.bat                   # 1-click Windows runner
```

---

## Standards Compliance

FruitVision AI adheres to:
- **ISO 22000 / HACCP**: Zero tolerance for severe necrotic rot in fresh market produce streams.
- **Codex Alimentarius (CODEX STAN 1995)**: Standards for fresh fruit defect categorization and primary vs secondary grading.
