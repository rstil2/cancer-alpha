# Master Installation Guide

This guide provides instructions for installing and setting up the Cancer Alpha system for various deployment scenarios.

## Prerequisites

- Docker installed on your system
- Python 3.8+ installed
- Access to a terminal with Git
- Basic understanding of command line operations

## Installation Steps

1. **Clone Repository**
   ```bash
   git clone https://github.com/rstil2/cancer-alpha.git
   cd cancer-alpha
   ```

2. **Install Python Dependencies**
   ```bash
   pip install -r DEMO_PACKAGE/cancer_genomics_ai_demo/requirements_streamlit.txt
   ```

3. **Build Docker Images (Optional)**
   ```bash
   docker build -t cancer-alpha:latest .
   ```

4. **Run Streamlit Demo**
   ```bash
   ./start_streamlit.sh
   ```

5. **Run Demo Server**
   ```bash
   python3 serve_demo.py
   ```

## Troubleshooting

- Ensure all dependencies are correctly installed.
- Check your network settings if you cannot access services.

For more detailed help, review the [Comprehensive Deployment Guide](COMPREHENSIVE_DEPLOYMENT_GUIDE.md).
