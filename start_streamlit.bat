@echo off

REM Cancer Alpha - Streamlit Demo Launcher (Windows)
REM This is a demonstration version using simplified models and synthetic data
REM For the full production system, contact craig.stillwell@gmail.com for licensing

echo 🧬 Cancer Alpha - Streamlit Demo Interface
echo ==========================================
echo.
echo ⚠️  DEMO VERSION NOTICE:
echo This is a demonstration using simplified models and synthetic data.
echo The full production system achieves 99.5%% accuracy with real genomic data.
echo.

REM Check if we're in the right directory
if not exist "DEMO_PACKAGE\cancer_genomics_ai_demo" (
    echo ❌ Error: DEMO_PACKAGE not found.
    echo Please make sure you're running this script from the cancer-alpha root directory.
    pause
    exit /b 1
)

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Error: Python 3 is required but not installed.
    echo Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

echo 🚀 Starting Cancer Alpha Streamlit Demo...
echo 📦 Using demo package with synthetic data
echo.

REM Navigate to demo directory
cd DEMO_PACKAGE\cancer_genomics_ai_demo

REM Check if requirements are installed
if not exist "requirements_installed.flag" (
    echo 📦 Installing demo requirements...
    pip install -r requirements_streamlit.txt
    echo. > requirements_installed.flag
)

echo 🌐 Starting Streamlit interface...
echo 📍 Open your browser to: http://localhost:8501
echo 🔍 Features: Random Forest model, SHAP explainability, synthetic data
echo.
echo Press Ctrl+C to stop the demo
echo.

REM Start the Streamlit app
python -m streamlit run streamlit_app.py --server.port 8501 --server.address localhost
