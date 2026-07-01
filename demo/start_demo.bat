@echo off
setlocal
cd /d "%~dp0"

echo Oncura demo - starting Streamlit on http://localhost:8501
echo Research only. Not for clinical use.
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.8+ is required. Install from https://www.python.org/downloads/
    pause
    exit /b 1
)

if not exist "models\multimodal_real_tcga_random_forest.pkl" (
    echo Missing demo models. Re-download the full Oncura-Demo.zip package.
    pause
    exit /b 1
)

echo Installing dependencies...
python -m pip install -q -r requirements_streamlit.txt
if errorlevel 1 (
    echo Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo Open http://localhost:8501 in your browser. Press Ctrl+C to stop.
echo.

python -m streamlit run streamlit_app.py --server.port 8501 --server.address localhost
pause
