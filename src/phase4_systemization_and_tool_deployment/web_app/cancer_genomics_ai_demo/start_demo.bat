@echo off
echo 🧬 Cancer Genomics AI Classifier - Demo
echo =====================================
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
echo 📦 Installing dependencies...
pip install -r requirements_streamlit.txt

echo 🧪 Testing setup...
python test_models.py

if errorlevel 1 (
    echo ❌ Setup test failed. Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo 🚀 Starting Cancer Genomics AI Classifier...
echo 🌐 Open your browser to: http://localhost:8501
echo ⏹️  Press Ctrl+C to stop
echo.

streamlit run streamlit_app.py --server.port 8501 --server.address localhost
pause
