# Launch script for PowerShell
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "   Starting HIN-GNN News Recommendation Servers (Frontend + Backend)  " -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan

Set-Location -Path $PSScriptRoot

Write-Host "`n[1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", ".\venv\Scripts\python.exe api.py" -WindowStyle Minimized

Write-Host "[2/2] Starting Streamlit Frontend on http://localhost:8501 ..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", ".\venv\Scripts\streamlit.exe run app.py --server.port 8501" -WindowStyle Minimized

Write-Host "`n======================================================================" -ForegroundColor Cyan
Write-Host " Servers are now running! Open these links in your browser:" -ForegroundColor Green
Write-Host " 1. Frontend UI:   http://localhost:8501" -ForegroundColor White
Write-Host " 2. Admin Panel:   http://127.0.0.1:8000/dashboard" -ForegroundColor White
Write-Host " 3. Swagger Docs:  http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host "======================================================================" -ForegroundColor Cyan

