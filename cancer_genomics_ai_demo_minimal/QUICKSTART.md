# 🚀 Quick Start Guide

## 30-Second Demo Setup

### Prerequisites
- Python 3.8+ installed
- 4GB+ RAM available

### Installation (3 commands)

1. **Download & Navigate**
   ```bash
   git clone [YOUR_REPO_URL]
   cd cancer_genomics_ai_demo_minimal
   ```

2. **Setup & Install**
   ```bash
   python setup.py
   ```

3. **Launch Demo**
   ```bash
   # macOS/Linux:
   ./start_demo.sh
   
   # Windows:
   start_demo.bat
   ```

4. **Open Browser**: http://localhost:8501

---

## ⚡ First Demo Run

1. **Select "Sample Data"** in sidebar
2. **Choose "Cancer Sample"**
3. **Click "Generate Sample Data"**
4. **View Results**: Cancer type prediction with 95% accuracy model
5. **Explore SHAP explanations** for model interpretability

---

## 🎯 What You'll See

- **🏆 Production LightGBM + SMOTE (95.0% accuracy)** - Best model
- **Multi-modal genomic features** (110 features across 6 data types)
- **Real-time predictions** with confidence scores
- **SHAP explanations** showing which features drive predictions
- **Interactive visualizations** with biological insights

---

## 🆘 Quick Troubleshooting

**Port 8501 in use?**
```bash
streamlit run streamlit_app.py --server.port 8502
```

**Missing dependencies?**
```bash
pip install -r requirements_streamlit.txt
```

**Still issues?** Contact: craig.stillwell@gmail.com

---

## ⚠️ Important Notes

- **For demonstration only** - not for medical diagnosis
- **Patent protected technology** (Application No. 63/847,316)
- All processing happens locally on your machine
- No genomic data is stored or transmitted

---

*🧬 Experience the future of AI-powered cancer genomics!*
