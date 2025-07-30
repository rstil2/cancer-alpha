# Models and Data Download Instructions

This lightweight demo package contains all the source code and documentation, but excludes large model files and datasets to keep the package size manageable for GitHub distribution.

## What's Included in This Package

✅ **Complete source code** - All training, inference, and API code  
✅ **Documentation** - README files, progress logs, and technical documentation  
✅ **Configuration files** - Docker, Kubernetes, and deployment configurations  
✅ **Scripts** - Shell scripts for setup and execution  
✅ **API implementation** - FastAPI backend and Streamlit frontend  
✅ **Test suites** - Comprehensive validation and testing code  

## What's NOT Included (Available Separately)

❌ **Trained models** (~3.9GB) - PyTorch model files (.pth, .pkl)  
❌ **TCGA datasets** (~2.3GB) - Raw and processed genomic data  
❌ **Cache files** - Intermediate processing results  

## How to Get the Full Functionality

### Option 1: Train Your Own Models (Recommended for Learning)
```bash
# Install dependencies
pip install -r requirements_streamlit.txt

# Download TCGA data (will create data_integration/tcga_large_cache/)
python scalable_tcga_downloader.py

# Train models (will create models/ directory)
python train_real_tcga_90_percent.py
```

### Option 2: Download Pre-trained Models and Data
If you need the pre-trained models and processed datasets, they are available as separate downloads:

**Models Package** (~3.9GB)
- Contains all trained PyTorch models (.pth files)
- Includes scalers and preprocessors (.pkl files) 
- Performance results and training logs

**Data Package** (~2.3GB)  
- Processed TCGA genomic data
- Mutation, methylation, and copy number files
- Clinical metadata

**Contact Information**: For access to the full models and data packages, please reach out through the project repository or create an issue requesting download links.

### Option 3: Use Docker (Models Downloaded Automatically)
```bash
# Build and run with Docker - will download data as needed
docker-compose up --build
```

## Quick Start (Code Only)

You can explore the codebase and run basic functionality without the large files:

```bash
# Extract the package
unzip cancer_genomics_ai_demo_lightweight.zip
cd cancer_genomics_ai_demo/

# Install dependencies  
pip install -r requirements_streamlit.txt

# Run tests (using synthetic data)
python test_demo_comprehensive.py

# Start the API server (limited functionality without models)
cd api && python main.py

# View documentation
cat README.md
cat PROJECT_PROGRESS_LOG.md
```

## File Structure

```
cancer_genomics_ai_demo/
├── README.md                          # Main documentation
├── streamlit_app.py                   # Web interface
├── api/                               # FastAPI backend
├── src/                               # Organized source code
├── scripts/                           # Automation scripts  
├── tests/                             # Test suites
├── models/                            # [EMPTY] - Download separately
├── data_integration/tcga_large_cache/ # [EMPTY] - Download separately
└── data/                              # [EMPTY] - Download separately
```

## Development Workflow

1. **Start with lightweight package** - Explore code and documentation
2. **Download sample data** - Use `scalable_tcga_downloader.py` to get small dataset
3. **Train small model** - Use `train_enhanced_model.py` for quick testing
4. **Scale up** - Download full datasets and pre-trained models as needed

## Why This Approach?

- **GitHub Friendly**: Keeps repository under size limits
- **Fast Downloads**: Quick to get started with the codebase  
- **Flexible**: Choose your own data and model scope
- **Educational**: Encourages understanding the training pipeline
- **Production Ready**: Full functionality available when needed

## Support

For questions about downloading the full models and data packages, please:
1. Check the main repository README for latest instructions
2. Create an issue with your specific needs
3. Contact the maintainers for enterprise access

---

**Note**: The lightweight package is fully functional for code exploration, development, and small-scale testing. The additional downloads are only needed for full-scale inference with production models or working with the complete TCGA dataset.
