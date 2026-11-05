# FruitVision AI — Intelligent Fruit Quality Inspection & Sorting System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI%200.110-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch%20%7C%20EfficientNetB0-EE4C2C.svg)](https://pytorch.org/)
[![React 19](https://img.shields.io/badge/Frontend-React%2019%20%2B%20Tailwind%20v4-61DAFB.svg)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An Industry 4.0 automated fruit quality inspection, defect analysis, and conveyor sorting simulation system powered by **Transfer-Learned EfficientNetB0**, OpenCV computer vision segmentation, and a strict monochrome industrial control room UI.

---

## System Architecture & Features

- **Transfer-Learned EfficientNetB0**: Pretrained on ImageNet and fine-tuned for tri-class produce grading (`Good`, `Damaged`, `Rotten`).
- **Out-of-Distribution "Not Fruits" Detector**: Multi-layer optical rejection engine that identifies non-produce artifacts (electronics, documents, vehicles, blanks) and flags them as `[NOT FRUITS]`.
- **Interactive Conveyor Belt Simulation**: Real-time animated sorting belt with optical laser scanner, pneumatic divert gates (`GATE_1_ACCEPTED`, `GATE_2_SECONDARY`, `GATE_3_REJECTED`), live bin telemetry, and speed toggles (1x, 2x, 4x).
- **Multi-Fruit Batch Segmentation**: OpenCV contour detection and individual fruit isolation for conveyor batch processing.
- **Real-Time Webcam Monitor**: Live video feed processing with bounding-box HUD overlay and automatic defect logging.
- **Persistent SQLite Audit Logging**: Complete historical traceability with indexed search, status filters, CSV export, and annotated visual evidence snapshots.
- **Telemetry & Quality Analytics**: 7-day acceptance trajectory line charts, defect distribution breakdowns, and held-out model confusion matrix.

---

## Prerequisites

- **Python**: Version `3.10` or higher
- **Node.js**: Version `18.0` or higher (with `npm`)
- **Git**

---

## Commands to Run the Project

### Option A: One-Click Quick Start (Windows)

Run the included automated launcher in the repository root:

```cmd
start.bat
```

This automatically starts:
- **FastAPI Backend Server** on `http://127.0.0.1:8001`
- **Vite React Frontend** on `http://localhost:5174`

---

### Option B: Manual Setup & Execution

#### 1. Clone the Repository
```bash
git clone https://github.com/ayushgatla/FruitVision.git
cd FruitVision
```

#### 2. Backend Setup & Startup
Open a terminal in the project root directory:

```bash
# Create a Python virtual environment (optional but recommended)
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
# source venv/bin/activate

# Install backend dependencies
pip install -r backend/requirements.txt

# Start the FastAPI backend server (Runs on Port 8001)
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8001 --reload
```

> **Backend Verification**:
> - API Root / Health Check: [http://127.0.0.1:8001/api/health](http://127.0.0.1:8001/api/health)
> - Interactive OpenAPI Swagger Docs: [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

---

#### 3. Frontend Setup & Startup
Open a **second terminal** in the `frontend` folder:

```bash
cd frontend

# Install Node.js dependencies
npm install

# Launch Vite development server (Runs on Port 5174)
npm run dev
```

> **Web Application Access**:
> Open your browser and navigate to **[http://localhost:5174](http://localhost:5174)**.

---

#### 4. Seed Realistic Testing Data (Optional)
To populate or reset the database with realistic industrial optical inspection batches spanning from October 25 (with 1-2 day operational skips in between):

```bash
python backend/scripts/seed_test_data.py
```

---

## Running Model Training & Evaluation

### Local Model Calibration
To train or recalibrate the EfficientNetB0 classification head locally on sample images:

```bash
python -m training.train_local
```

### Held-Out Test Evaluation & Metric Curves
To evaluate model performance and regenerate confusion matrix and loss/accuracy curves:

```bash
python -m training.evaluate
```

Generated assets:
- `training/confusion_matrix.png`
- `training/loss_accuracy_curve.png`

### Full Kaggle GPU Training (Google Colab)
For high-volume GPU training on 18,000+ Kaggle fruit images (*Fruits-360* + *Fresh and Rotten Fruits* dataset):

1. Upload [`training/train.ipynb`](training/train.ipynb) to [Google Colab](https://colab.research.google.com/).
2. Set runtime type to **T4 GPU**.
3. Run all cells (`Ctrl + F9`) to download datasets, train the 2-stage transfer learning pipeline, and export `best.pt`.
4. Copy the resulting `best.pt` into the `weights/` directory and click **`[RELOAD]`** in the UI header.

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
├── frontend/
│   ├── src/
│   │   ├── components/         # Header, StatCards, HistoryTable
│   │   ├── pages/              # Dashboard, InspectionHub, ConveyorSimulation, LiveMonitor, History, Analytics
│   │   └── services/           # Axios API client
│   ├── package.json            # Node dependencies
│   └── vite.config.js          # Vite config & API reverse proxy (to :8001)
├── sample_images/              # Presets (good apples, oranges, bananas, rotten, non-fruit)
├── training/
│   ├── train.ipynb             # Google Colab GPU training notebook
│   ├── train_local.py          # Local PyTorch calibration script
│   └── evaluate.py             # Evaluation & confusion matrix script
├── weights/                    # Active model checkpoint (best.pt)
├── start.bat                   # 1-click Windows runner
└── README.md                   # Project documentation & run guide
```

---

## Standards Compliance

FruitVision AI adheres to:
- **ISO 22000 / HACCP**: Zero tolerance for severe necrotic rot in fresh market produce streams.
- **Codex Alimentarius (CODEX STAN 1995)**: Standards for fresh fruit defect categorization and primary vs secondary grading.
