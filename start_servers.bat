@echo off
title Launch HIN-GNN News Recommendation System
echo ======================================================================
echo    Starting HIN-GNN News Recommendation Servers (Frontend + Backend)
echo ======================================================================
echo.

cd /d "%~dp0"

echo [1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "HIN-GNN Backend API" /min cmd /c ".\venv\Scripts\python.exe api.py"

echo [2/2] Starting Streamlit Frontend on http://localhost:8501 ...
start "HIN-GNN Frontend UI" /min cmd /c ".\venv\Scripts\streamlit.exe run app.py --server.port 8501"

echo.
echo ======================================================================
echo  Servers are launching! You can open these links in your browser:
echo  1. Frontend UI:   http://localhost:8501
echo  2. Admin Panel:   http://127.0.0.1:8000/dashboard
echo  3. Swagger Docs:  http://127.0.0.1:8000/docs
echo ======================================================================
echo.
pause

