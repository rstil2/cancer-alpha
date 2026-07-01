@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

title Oncura Demo
echo.
echo  Oncura Demo
echo  ===========
echo  Research only - not for clinical use.
echo.

REM Prefer Windows Python launcher, then python on PATH
set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY where python >nul 2>&1 && set "PY=python"

if not defined PY (
    echo  Python 3.8+ is not installed or not on PATH.
    echo  Install from https://www.python.org/downloads/
    echo  On Windows, check "Add Python to PATH" during setup.
    echo.
    start "" "https://www.python.org/downloads/"
    pause
    exit /b 1
)

if not exist "models\multimodal_real_tcga_random_forest.pkl" (
    echo  Missing model files. Unzip the full Oncura-Demo folder and try again.
    pause
    exit /b 1
)

echo  Installing dependencies (first run may take a minute)...
%PY% -m pip install -q -r requirements_streamlit.txt
if errorlevel 1 (
    echo  pip install failed.
    pause
    exit /b 1
)

echo.
echo  Starting demo... your browser should open to http://localhost:8501
echo  Press Ctrl+C in this window to stop.
echo.

%PY% -m streamlit run streamlit_app.py --server.port 8501 --server.address localhost
pause
