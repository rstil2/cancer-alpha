# TCGA Data Directory

This directory will contain the downloaded and processed TCGA data after running the download and processing scripts.

## Expected Files After Download/Processing:

- `tcga_processed_data.npz` - Processed data for model training
- `expanded_real_tcga_data.npz` - Expanded dataset
- `multimodal_tcga_data.npz` - Multimodal data

## How to Get These Files:

1.  **Download and Process**: Run `python scalable_tcga_downloader.py` followed by `python process_real_tcga_data.py`
2.  **Download Pre-processed**: See `DOWNLOAD_MODELS_DATA.md` for download links to the data packages.
