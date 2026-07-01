# Master Installation Guide

This guide provides instructions for installing and setting up the Oncura system for various deployment scenarios.

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
   pip install -r requirements.txt
   cd cancer_genomics_ai_demo_minimal && pip install -r requirements_streamlit.txt
   ```

3. **Run Streamlit Demo**
   ```bash
   cd cancer_genomics_ai_demo_minimal
   python setup.py
   ./start_demo.sh
   ```

4. **Build Docker Images (Optional)**
   ```bash
   cd cancer_genomics_ai_demo_minimal && docker-compose up
   ```

## Troubleshooting

- Ensure all dependencies are correctly installed.
- Check your network settings if you cannot access services.

For more detailed help, review the [Comprehensive Deployment Guide](COMPREHENSIVE_DEPLOYMENT_GUIDE.md).
