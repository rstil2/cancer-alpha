#!/usr/bin/env python3
"""
Installation Verification Script for Cancer Genomics AI Demo
"""

import os
import sys
from pathlib import Path

def check_file_size(file_path):
    """Get human-readable file size"""
    size = os.path.getsize(file_path)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"

def verify_installation():
    """Verify all required files are present and correct"""
    
    print("🧬 Cancer Genomics AI Demo - Installation Verification")
    print("=" * 60)
    
    # Required files and their descriptions
    required_files = {
        "streamlit_app.py": "Main demo application",
        "usage_tracker.py": "Demo usage analytics",
        "requirements_streamlit.txt": "Python dependencies",
        "setup.py": "Setup script",
        "models/lightgbm_smote_production.pkl": "Production AI model",
        "models/standard_scaler.pkl": "Data preprocessing scaler",
    }
    
    # Optional files
    optional_files = {
        "models/model_metadata.json": "Model metadata",
        "start_demo.sh": "Unix start script",
        "start_demo.bat": "Windows start script",
        "Dockerfile": "Docker configuration",
        "docker-compose.yml": "Docker Compose setup",
    }
    
    print("📋 Required Files:")
    all_required_present = True
    
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            size = check_file_size(file_path)
            print(f"✅ {file_path:<40} ({size}) - {description}")
        else:
            print(f"❌ {file_path:<40} MISSING - {description}")
            all_required_present = False
    
    print("\n📋 Optional Files:")
    for file_path, description in optional_files.items():
        if os.path.exists(file_path):
            size = check_file_size(file_path)
            print(f"✅ {file_path:<40} ({size}) - {description}")
        else:
            print(f"⚪ {file_path:<40} Not present - {description}")
    
    print("\n" + "=" * 60)
    
    if all_required_present:
        print("🎉 All required files are present!")
        print("\n📖 Next steps:")
        print("1. Run: python setup.py")
        print("2. Start demo with: ./start_demo.sh (Unix) or start_demo.bat (Windows)")
        print("3. Open browser to: http://localhost:8501")
        return True
    else:
        print("❌ Missing required files. Please ensure all files are downloaded.")
        return False

def check_python_requirements():
    """Check if required Python packages are available"""
    print("\n🐍 Python Environment Check:")
    
    required_packages = [
        "streamlit",
        "pandas", 
        "numpy",
        "sklearn",
        "lightgbm",
        "shap",
        "plotly",
        "matplotlib"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Install missing packages with:")
        print("pip install -r requirements_streamlit.txt")
        return False
    else:
        print("\n🎉 All required packages are installed!")
        return True

def main():
    """Main verification function"""
    files_ok = verify_installation()
    packages_ok = check_python_requirements()
    
    print("\n" + "=" * 60)
    if files_ok and packages_ok:
        print("🚀 Ready to launch! Run: python setup.py")
    else:
        print("🔧 Setup needed. Follow the installation instructions.")
    
    print("\n⚠️  Remember: This is demonstration software only")
    print("   Patent: Provisional Application No. 63/847,316")
    print("   Contact: craig.stillwell@gmail.com")

if __name__ == "__main__":
    main()
