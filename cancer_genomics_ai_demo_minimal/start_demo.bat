@echo off
echo 🧬 Cancer Alpha - Production Demo (95.0%% Accuracy)
echo ================================================
echo.

echo 🏆 PRODUCTION DEMO PACKAGE:
echo Features our breakthrough LightGBM + SMOTE model:
echo - Production LightGBM + SMOTE (95.0%% accuracy)
echo - Multi-modal genomic data (110 features)
echo - Full SHAP explanations
echo - 8 cancer types classification
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python 3 is required but not installed
    echo    Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found

REM Check for required model files
echo 🔍 Checking for production models...
if not exist "models\lightgbm_smote_production.pkl" (
    echo ❌ Production model not found: models\lightgbm_smote_production.pkl
    echo    Please ensure you have the complete demo package.
    pause
    exit /b 1
)

if not exist "models\standard_scaler.pkl" (
    echo ❌ Scaler not found: models\standard_scaler.pkl
    echo    Please ensure you have the complete demo package.
    pause
    exit /b 1
)

echo ✅ Production models found

echo 📦 Installing dependencies...
pip install -r requirements_streamlit.txt

echo.
REM Clear any existing processes on port 8501
echo 🔄 Clearing port 8501...

REM First, kill any Streamlit processes
echo    Killing existing Streamlit processes...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq streamlit*" >nul 2>&1
taskkill /F /IM pythonw.exe /FI "WINDOWTITLE eq streamlit*" >nul 2>&1
timeout /t 2 >nul 2>&1

REM Then kill any processes using port 8501
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8501 2^>nul') do (
    echo    Killing existing process on port 8501 (PID: %%a)
    taskkill /PID %%a /F >nul 2>&1
)
timeout /t 3 >nul 2>&1

REM Final verification - kill any remaining processes on port 8501
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8501 2^>nul') do (
    echo    Final cleanup: Killing PID %%a
    taskkill /PID %%a /F >nul 2>&1
)
echo ✅ Port 8501 cleared

echo 🚀 Starting Cancer Genomics AI Classifier...
echo 🌐 Open your browser to: http://localhost:8501
echo ⏹️  Press Ctrl+C to stop
echo ⚠️  Note: This is demonstration software only - not for medical diagnosis
echo.

streamlit run streamlit_app.py --server.port 8501
pause
