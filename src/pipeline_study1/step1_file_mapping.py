#!/usr/bin/env python3
"""Study 1 Step 1: File-to-patient mapping (extends Study 2 map with KIRC)."""

import csv
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CANCER_TYPES, DATA_DIR, MAPPING_CSV, OUTPUT_DIR, PROJECT_ROOT

LEGACY_MAPPING = PROJECT_ROOT / "data" / "file_patient_mapping.csv"
GDC_FILES_URL = "https://api.gdc.cancer.gov/files"
BATCH_SIZE = 500
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
        except (requests.RequestException, KeyError) as exc:
            print(f"    Retry {attempt + 1}/3: {exc}")
            time.sleep(2 ** attempt)
    return {"hits": [], "pagination": {"total": 0}}


def get_all_mappings(project_id, data_type_key):
    data_type = DATA_TYPE_MAP[data_type_key]
    workflow = WORKFLOW_MAP.get(data_type_key)
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
        time.sleep(0.2)
    return mappings


def build_local_file_index():
    local_files = {}
    for modality in ["expression", "mutations", "methylation", "copy_number"]:
        modality_dir = DATA_DIR / modality
        if not modality_dir.exists():
            continue
        for cancer_dir in modality_dir.iterdir():
            if not cancer_dir.is_dir() or cancer_dir.name not in CANCER_TYPES:
                continue
            for f in cancer_dir.iterdir():
                if f.is_file():
                    local_files[f.name] = {
                        "local_path": str(f),
                        "cancer_type": cancer_dir.name,
                        "modality": modality,
                    }
    return local_files


def attach_local_paths(rows, local_files):
    for m in rows:
        local = local_files.get(m["file_name"], {})
        m["has_local_file"] = bool(local)
        m["local_path"] = local.get("local_path", "")
        m["local_cancer_type"] = local.get("cancer_type", "")
    return rows


def main():
    print("=" * 60)
    print("Study 1 Step 1: File-to-patient mapping")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    local_files = build_local_file_index()
    print(f"Indexed {len(local_files)} local Study-1-panel files")

    rows = []
    if LEGACY_MAPPING.exists():
        legacy = pd.read_csv(LEGACY_MAPPING)
        legacy = legacy[legacy["project_id"].isin(CANCER_TYPES)]
        legacy = legacy[~legacy["project_id"].eq("TCGA-LUSC")]
        rows.extend(legacy.to_dict("records"))
        print(f"Loaded {len(rows)} rows from legacy mapping (excl. LUSC)")

    print("\nFetching TCGA-KIRC from GDC...")
    for data_type_key in DATA_TYPE_MAP:
        rows.extend(get_all_mappings("TCGA-KIRC", data_type_key))

    rows = attach_local_paths(rows, local_files)

    fieldnames = [
        "file_name", "file_id", "case_submitter_id", "project_id",
        "sample_submitter_id", "sample_type", "data_type",
        "has_local_file", "local_path", "local_cancer_type",
    ]
    with open(MAPPING_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    local_primary = [
        r for r in rows
        if r.get("has_local_file") and r.get("sample_type") == "Primary Tumor"
    ]
    print(f"\nWrote {len(rows)} mappings -> {MAPPING_CSV}")
    print(f"Local Primary Tumor files: {len(local_primary)}")
    for ct in CANCER_TYPES:
        n = sum(1 for r in local_primary if r["project_id"] == ct)
        print(f"  {ct}: {n}")


if __name__ == "__main__":
    main()
