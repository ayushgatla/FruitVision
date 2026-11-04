@echo off
echo =========================================================================
echo  FruitVision AI -- Intelligent Fruit Quality Inspection & Sorting System
echo =========================================================================
echo.

echo [1/2] Starting FastAPI Backend on http://127.0.0.1:8001...
start cmd /k "python -m backend.main"

timeout /t 3 /nobreak >nul

echo [2/2] Starting Vite React Frontend on http://localhost:5174...
cd frontend
start cmd /k "npm run dev"

echo.
echo =========================================================================
echo  FruitVision AI System is now running!
echo  Frontend Dashboard : http://localhost:5174
echo  Backend Swagger API: http://127.0.0.1:8001/docs
echo =========================================================================

