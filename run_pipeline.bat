@echo off
REM ============================================================
REM  HIN-GNN News Recommendation System — Full Run Script
REM  Generates dataset, trains models, evaluates, saves artifacts
REM ============================================================

echo.
echo ============================================================
echo   HIN-GNN + HAN News Recommendation Pipeline
echo ============================================================
echo.

REM Use venv python (handles path with spaces via cd first)
cd /d "%~dp0"

echo [1/3] Generating MIND-small dataset...
venv\Scripts\python.exe data\mind_small_generator.py
if %errorlevel% neq 0 (
    echo ERROR: Dataset generation failed!
    pause
    exit /b 1
)

echo.
echo [2/3] Running training and evaluation pipeline...
venv\Scripts\python.exe pipeline.py
if %errorlevel% neq 0 (
    echo ERROR: Pipeline failed!
    pause
    exit /b 1
)

echo.
echo [3/3] Done! Check artifacts\ for results.
echo.
echo   - artifacts\results.csv           (model comparison)
echo   - artifacts\diversity_tradeoff.csv (alpha sweep)
echo   - artifacts\comparison_bar.png    (bar chart)
echo   - artifacts\tradeoff_plot.png     (accuracy-diversity)
echo   - artifacts\embeddings.npy        (HIN-GNN embeddings)
echo   - artifacts\han_embeddings.npy    (HAN embeddings)
echo.
pause

