# FruitVision AI — Presentation Quick Reference Cheat Sheet

> **Keep this one-page reference open during your demonstration.**  
> All commands use verified paths in `C:\Users\ramsa\Desktop\FruitVision`.

---

## 1. Quick Startup Commands (PowerShell)

### Terminal 1: FastAPI Backend
```powershell
cd C:\Users\ramsa\Desktop\FruitVision
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8001 --reload
```
*Expected: `[FruitVision AI] Ready. Device: CPU. Custom weights active: True`*

### Terminal 2: React Frontend
```powershell
cd C:\Users\ramsa\Desktop\FruitVision\frontend
npm.cmd run dev
```
*Expected: `VITE ready in ~500ms on http://localhost:5174/`*

---

## 2. Essential Local URLs

| Service | Port | Local URL | Description |
| :--- | :---: | :--- | :--- |
| **Frontend UI** | `5174` | **http://localhost:5174** | Main Industrial Control Dashboard |
| **Backend API** | `8001` | **http://127.0.0.1:8001** | FastAPI Server Root |
| **Interactive Docs** | `8001` | **http://127.0.0.1:8001/docs** | Interactive OpenAPI / Swagger Spec |
| **System Health** | `8001` | **http://127.0.0.1:8001/api/health** | Real-time Device & Model Status JSON |

---

## 3. Verified Sample Images for Demo

| Fruit Target | Filename | Preset Button in UI | Expected Result |
| :--- | :--- | :--- | :--- |
| **Fresh Apple** | `sample_good_apple.jpg` | `[FRESH APPLE]` | `ACCEPTED` (Good, Conf: 51.5%) |
| **Rotten Apple** | `sample_rotten_apple.jpg` | `[ROTTEN APPLE]` | `REJECTED` (Rotten, Conf: 99.5%) |
| **Fresh Banana** | `sample_good_banana.jpg` | Manual upload | `ACCEPTED` (Good, Conf: 96.3%) |
| **Rotten Banana** | `sample_rotten_banana.jpg` | `[ROTTEN BANANA]` | `REJECTED` (Rotten, Conf: 99.5%) |
| **Damaged Apple** | `sample_damaged_apple.jpg` | `[DAMAGED APPLE]` | `SECONDARY_INSPECTION` (Damaged, Conf: 88.1%) |
| **Non-Produce Item** | `sample_not_fruit.jpg` | `[NOT FRUIT (TEST)]` | `NON_PRODUCE` (Rejected: Blue/Cyan Hue) |
| **Multi-Fruit Batch** | `sample_multi_fruits.jpg` | `[LOAD SAMPLE MULTI-FRUIT]` | 4 Bounding Boxes: 2 Good, 1 Damaged, 1 Rotten |

*Physical file location on disk:* `C:\Users\ramsa\Desktop\FruitVision\sample_images\`

---

## 4. Main 5-Minute Presentation Sequence

1. **Dashboard (`http://localhost:5174`):**  
   Introduce FruitVision AI as an Industry 4.0 fruit inspection & automated sorting system. Show top KPIs (Acceptance Rate, Defect Count, Active Alerts).
2. **Optical Hub (`INSPECT` Tab):**  
   - Click `[FRESH APPLE]` preset $\rightarrow$ Show Grade A Accepted (Green HUD banner).
   - Click `[ROTTEN APPLE]` preset $\rightarrow$ Show Bio-Waste Rejection (Red HUD banner, ISO 22000 zero-tolerance).
   - Click `[NOT FRUIT (TEST)]` preset $\rightarrow$ Show Optical Out-of-Distribution Rejection.
3. **Multi-Fruit Batch (`[MULTI-FRUIT BATCH]` Sub-Tab):**  
   Click `[LOAD SAMPLE MULTI-FRUIT BATCH]` $\rightarrow$ Show OpenCV contour detection isolating 4 individual fruit crops with bounding boxes.
4. **Conveyor Sorting (`CONVEYOR` Tab):**  
   - Select `LINE: [APPLE DEDICATED]` $\rightarrow$ Click `[STEP]` $\rightarrow$ Show Gate 1 / Gate 2 / Gate 3 pneumatic diverting.
   - Switch to `LINE: [ALL PRODUCE MIX]` $\rightarrow$ Click `[START SIMULATION]` $\rightarrow$ Show live sorting speed and counter incrementing.
5. **Audit History (`HISTORY` Tab):**  
   Show real-time SQLite logs, search for `"Rotten"`, and click `[EXPORT CSV]` (RFC 4180 formula injection safe).
6. **Analytics (`ANALYTICS` Tab):**  
   Point out 100% held-out test accuracy, precision/recall table, and the $4 \times 4$ confusion matrix.

---

## 5. Instant Error Fixes

| Issue Encountered | Quick PowerShell Fix |
| :--- | :--- |
| **PowerShell script execution blocked** | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force` |
| **Port 8001 already in use** | `Stop-Process -Id (Get-NetTCPConnection -LocalPort 8001).OwningProcess -Force` |
| **Port 5174 already in use** | `Stop-Process -Name node -Force` then run `npm.cmd run dev` |
| **`npm` not recognized or blocked** | Use `npm.cmd run dev` instead of `npm run dev` |
| **Frontend shows "OFFLINE"** | Ensure Terminal 1 (Backend) is started and running on port 8001. |
| **Empty History / Charts** | Run 3 steps in Conveyor or click 2 sample presets to populate SQLite. |

---

## 6. Clean Shutdown Command

- **In Terminal 2 (Frontend):** Press `Ctrl + C`, then press `Y` and `Enter`.
- **In Terminal 1 (Backend):** Press `Ctrl + C`. (Uvicorn exits cleanly).
- **Data Persistence:** All records in `backend/fruitvision.db` remain safely preserved for your next session.

