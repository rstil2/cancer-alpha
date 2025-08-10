#!/usr/bin/env python3
"""
Setup script for Cancer Genomics AI Demo
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    else:
        print(f"✅ Python version check passed: {sys.version.split()[0]}")

def install_requirements():
    """Install required packages"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements_streamlit.txt"
        ])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        sys.exit(1)

def check_model_files():
    """Check if required model files exist"""
    required_files = [
        "models/lightgbm_smote_production.pkl",
        "models/standard_scaler.pkl",
        "streamlit_app.py",
        "usage_tracker.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        sys.exit(1)
    else:
        print("✅ All required files found")

def create_start_script():
    """Create platform-specific start script"""
    system = platform.system()
    
    if system in ["Darwin", "Linux"]:  # macOS or Linux
        script_content = """#!/bin/bash
echo "🧬 Starting Cancer Genomics AI Demo..."
echo "🌐 Demo will be available at: http://localhost:8501"
echo "⚠️  Note: This is demonstration software only - not for medical diagnosis"
echo ""
streamlit run streamlit_app.py
"""
        with open("start_demo.sh", "w") as f:
            f.write(script_content)
        os.chmod("start_demo.sh", 0o755)
        print("✅ Created start_demo.sh")
        
    elif system == "Windows":
        script_content = """@echo off
echo 🧬 Starting Cancer Genomics AI Demo...
echo 🌐 Demo will be available at: http://localhost:8501
echo ⚠️  Note: This is demonstration software only - not for medical diagnosis
echo.
streamlit run streamlit_app.py
"""
        with open("start_demo.bat", "w") as f:
            f.write(script_content)
        print("✅ Created start_demo.bat")

def main():
    """Main setup function"""
    print("🧬 Cancer Genomics AI Demo - Setup")
    print("=" * 50)
    print("⚠️  PATENT PROTECTED TECHNOLOGY")
    print("   Patent: Provisional Application No. 63/847,316")
    print("   Contact: craig.stillwell@gmail.com")
    print("=" * 50)
    print()
    
    # Check Python version
    check_python_version()
    
    # Check for required files
    check_model_files()
    
    # Install dependencies
    install_requirements()
    
    # Create start script
    create_start_script()
    
    print()
    print("🎉 Setup completed successfully!")
    print()
    print("📖 To start the demo:")
    
    system = platform.system()
    if system in ["Darwin", "Linux"]:
        print("   ./start_demo.sh")
    elif system == "Windows":
        print("   start_demo.bat")
    else:
        print("   streamlit run streamlit_app.py")
    
    print()
    print("🌐 The demo will be available at: http://localhost:8501")
    print("⚠️  Remember: This is for demonstration only - not for medical diagnosis")

if __name__ == "__main__":
    main()
