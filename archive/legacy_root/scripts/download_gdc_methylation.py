"""
Batch Download SeSAMe Methylation Files from GDC API

- Downloads methylation files for specified cancer types
- Uses GDC API to query and fetch file UUIDs
- Saves manifest and downloads files using GDC Data Transfer Tool
"""

import requests
import os
import json
from pathlib import Path

# Cancer types to target
CANCER_TYPES = ["BRCA", "LUAD", "COAD", "PRAD", "STAD", "HNSC", "LUSC", "LIHC"]
DATA_TYPE = "Methylation"
PLATFORM = "Illumina Human Methylation 450"
LEVEL = "3"

MANIFEST_PATH = Path("data/gdc_methylation_manifest.txt")
DOWNLOAD_DIR = Path("data/gdc_methylation_download/")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# GDC API endpoint
API_URL = "https://api.gdc.cancer.gov/files"

# Build filters for query
filters = {
    "op": "and",
    "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": [f"TCGA-{ct}" for ct in CANCER_TYPES]}},
        {"op": "in", "content": {"field": "data_type", "value": [DATA_TYPE]}},
        {"op": "in", "content": {"field": "platform", "value": [PLATFORM]}},
        {"op": "in", "content": {"field": "data_category", "value": ["DNA Methylation"]}},
        {"op": "in", "content": {"field": "experimental_strategy", "value": ["Methylation Array"]}},
        {"op": "in", "content": {"field": "access", "value": ["open"]}},
        {"op": "in", "content": {"field": "file_format", "value": ["txt", "tsv"]}},
        {"op": "in", "content": {"field": "data_type", "value": ["Methylation Beta Value"]}},
        {"op": "in", "content": {"field": "data_level", "value": [LEVEL]}},
    ]
}

params = {
    "filters": json.dumps(filters),
    "fields": "file_id,file_name,cases.submitter_id,cases.project.project_id",
    "format": "JSON",
    "size": "10000"
}

print("Querying GDC API for methylation files...")
response = requests.get(API_URL, params=params)
response.raise_for_status()
data = response.json()

# GDC returns file list under 'data' or 'hits' depending on endpoint
if "data" in data and isinstance(data["data"], dict) and "hits" in data["data"]:
    hits = data["data"]["hits"]
elif "data" in data and isinstance(data["data"], list):
    hits = data["data"]
else:
    hits = []

file_ids = [f["file_id"] for f in hits]
print(f"Found {len(file_ids)} methylation files.")

# Write manifest for GDC Data Transfer Tool
gdc_manifest = {
    "ids": file_ids
}

manifest_url = "https://api.gdc.cancer.gov/data"

with open(MANIFEST_PATH, "w") as f:
    for fid in file_ids:
        f.write(f"{fid}\n")

print(f"Manifest written to {MANIFEST_PATH}")
print("To download, run:")
print(f"gdc-client download -m {MANIFEST_PATH} -d {DOWNLOAD_DIR}")
