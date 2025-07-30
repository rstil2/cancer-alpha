# Models Directory

This directory will contain the trained PyTorch models after you download them or train them yourself.

## Expected Files After Download/Training:

### Transformer Models (PyTorch .pth files)
- `real_tcga_90_transformer.pth` - Main 90% accuracy model
- `ultra_tcga_near_100_transformer.pth` - Near-100% accuracy model  
- `multimodal_transformer.pth` - Multimodal integration model
- Various checkpoint files (`checkpoint_acc_*.pth`)

### Scalers and Preprocessors (.pkl files)
- `real_tcga_90_scaler.pkl` - Data scaler for 90% model
- `multimodal_real_tcga_scaler.pkl` - Scaler for multimodal data
- `scaler.pkl` - General purpose scaler

### Traditional ML Models (.pkl files)
- `random_forest_model.pkl` - Random Forest baseline
- `gradient_boosting_model_new.pkl` - Gradient Boosting model
- `deep_neural_network_model_new.pkl` - Standard neural network

### Results and Metadata (.json files)
- `real_tcga_90_results.json` - Performance metrics
- `multimodal_real_tcga_training_report.json` - Training logs
- `hyperparameter_optimization.json` - Optimization results

## How to Get These Files:

1. **Train yourself**: Run `python train_real_tcga_90_percent.py`
2. **Download pre-trained**: See `DOWNLOAD_MODELS_DATA.md` for download links
3. **Docker**: Models will be downloaded automatically in containerized setup

## Size Information:
Total models directory: ~3.9GB when fully populated
