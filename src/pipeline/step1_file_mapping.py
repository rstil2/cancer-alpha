#!/usr/bin/env python3
"""Step 1: Build file-to-patient mapping via GDC API.

Queries GDC API in batches to map local file UUIDs (from filenames) to
TCGA patient barcodes. Covers expression, mutations, methylation, and
copy number files for 8 target cancer types.
"""

import csv
import json
import os
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "production_tcga"
OUTPUT_CSV = PROJECT_ROOT / "data" / "file_patient_mapping.csv"

CANCER_TYPES = [
    "TCGA-BRCA", "TCGA-LUAD", "TCGA-COAD", "TCGA-PRAD",
    "TCGA-STAD", "TCGA-HNSC", "TCGA-LUSC", "TCGA-LIHC",
]

GDC_FILES_URL = "https://api.gdc.cancer.gov/files"
BATCH_SIZE = 500  # GDC API handles up to ~1000 per request
FIELDS = (
    "file_id,file_name,"
    "cases.submitter_id,cases.project.project_id,"
    "cases.samples.submitter_id,cases.samples.sample_type,"
    "data_type,experimental_strategy"
)

DATA_TYPE_MAP = {
    "expression": "Gene Expression Quantification",
    "mutations": "Masked Somatic Mutation",
    "methylation": "Methylation Beta Value",
    "copy_number": "Copy Number Segment",
}

WORKFLOW_MAP = {
    "expression": "STAR - Counts",
    "mutations": "Aliquot Ensemble Somatic Variant Merging and Masking",
    "methylation": "SeSAMe Methylation Beta Estimation",
    "copy_number": "GATK4 MuTect2 Annotation",
}


def query_gdc_batch(project_id, data_type, workflow_type, offset=0, size=500):
    """Query GDC API for files matching project, data type, and workflow."""
    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": project_id}},
            {"op": "=", "content": {"field": "data_type", "value": data_type}},
        ],
    }
    if workflow_type:
        filters["content"].append(
            {"op": "=", "content": {"field": "analysis.workflow_type", "value": workflow_type}}
        )

    params = {
        "filters": json.dumps(filters),
        "fields": FIELDS,
        "size": size,
        "from": offset,
    }

    for attempt in range(3):
        try:
            r = requests.get(GDC_FILES_URL, params=params, timeout=60)
            r.raise_for_status()
            return r.json()["data"]
        except (requests.RequestException, KeyError) as e:
            print(f"    Retry {attempt+1}/3: {e}")
            time.sleep(2 ** attempt)
    return {"hits": [], "pagination": {"total": 0}}


def get_all_mappings(project_id, data_type_key):
    """Get all file-to-patient mappings for a project and data type."""
    data_type = DATA_TYPE_MAP[data_type_key]
    workflow = WORKFLOW_MAP.get(data_type_key)

    # First query to get total count
    result = query_gdc_batch(project_id, data_type, workflow, offset=0, size=1)
    total = result["pagination"]["total"]
    print(f"  {data_type_key}: {total} files in GDC")

    mappings = []
    offset = 0
    while offset < total:
        batch = query_gdc_batch(project_id, data_type, workflow, offset=offset, size=BATCH_SIZE)
        for hit in batch["hits"]:
            if not hit.get("cases"):
                continue
            case = hit["cases"][0]
            sample_type = "Unknown"
            sample_submitter = ""
            if case.get("samples"):
                sample_type = case["samples"][0].get("sample_type", "Unknown")
                sample_submitter = case["samples"][0].get("submitter_id", "")

            mappings.append({
                "file_name": hit["file_name"],
                "file_id": hit["file_id"],
                "case_submitter_id": case["submitter_id"],
                "project_id": case.get("project", {}).get("project_id", project_id),
                "sample_submitter_id": sample_submitter,
                "sample_type": sample_type,
                "data_type": data_type_key,
            })
        offset += BATCH_SIZE
        time.sleep(0.3)  # Rate limiting

    return mappings


def build_local_file_index():
    """Index all local files by filename for quick lookup."""
    local_files = {}
    for modality in ["expression", "mutations", "methylation", "copy_number"]:
        modality_dir = DATA_DIR / modality
        if not modality_dir.exists():
            continue
        for cancer_dir in modality_dir.iterdir():
            if not cancer_dir.is_dir():
                continue
            cancer_type = cancer_dir.name
            for f in cancer_dir.iterdir():
                if f.is_file():
                    local_files[f.name] = {
                        "local_path": str(f),
                        "cancer_type": cancer_type,
                        "modality": modality,
                    }
    return local_files


def main():
    print("=" * 60)
    print("Step 1: Building file-to-patient mapping via GDC API")
    print("=" * 60)

    # Index local files
    print("\nIndexing local files...")
    local_index = build_local_file_index()
    print(f"  Found {len(local_index)} local files")

    # Query GDC for each cancer type and data type
    all_mappings = []
    for cancer_type in CANCER_TYPES:
        print(f"\n{cancer_type}:")
        for modality in ["expression", "mutations", "methylation", "copy_number"]:
            mappings = get_all_mappings(cancer_type, modality)

            # Match with local files
            matched = 0
            for m in mappings:
                if m["file_name"] in local_index:
                    m["local_path"] = local_index[m["file_name"]]["local_path"]
                    m["has_local_file"] = True
                    matched += 1
                else:
                    m["local_path"] = ""
                    m["has_local_file"] = False

            all_mappings.extend(mappings)
            print(f"    -> {matched}/{len(mappings)} matched to local files")

    # Write CSV
    print(f"\nWriting {len(all_mappings)} mappings to {OUTPUT_CSV}")
    fieldnames = [
        "file_name", "file_id", "case_submitter_id", "project_id",
        "sample_submitter_id", "sample_type", "data_type",
        "local_path", "has_local_file",
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_mappings)

    # Summary
    local_matched = sum(1 for m in all_mappings if m.get("has_local_file"))
    primary_tumor = sum(1 for m in all_mappings if m.get("sample_type") == "Primary Tumor" and m.get("has_local_file"))
    print(f"\nSummary:")
    print(f"  Total GDC mappings: {len(all_mappings)}")
    print(f"  Matched to local files: {local_matched}")
    print(f"  Primary Tumor with local files: {primary_tumor}")
    print("Done!")


if __name__ == "__main__":
    main()
