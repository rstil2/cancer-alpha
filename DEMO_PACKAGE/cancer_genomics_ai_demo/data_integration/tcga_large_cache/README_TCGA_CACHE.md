# TCGA Large Cache Directory

This directory will contain the raw TCGA data files downloaded from the NIH/NCI databases.

## Expected Subdirectories After Download:

- `mutations/` - Mutation annotation files (.maf.gz)
- `methylation/` - DNA methylation data files (.txt)
- `copy_number/` - Copy number variation files (.txt, .seg)
- `clinical/` - Clinical metadata files (.xml)

## How to Populate:

Run `python scalable_tcga_downloader.py` to download TCGA data directly from NIH/NCI.

## Size Information:
Total cache directory: ~2.3GB when fully populated with sample data
