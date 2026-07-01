#!/usr/bin/env python3
"""Study 1 Step 2: Extract per-modality features from real TCGA files."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    CLINICAL_CACHE,
    ICGC_EXPRESSION_GENES,
    MAPPING_CSV,
    MUTATION_GENES,
    OUTPUT_DIR,
    TOP_METHYLATION_PROBES,
)
from parsers import (
    fragmentomics_from_segments,
    icgc_expression_features,
    parse_clinical_xml,
    parse_copy_number_seg,
    parse_maf_features,
    parse_star_tpm,
)

PROBE_SAMPLE_PER_TYPE = 40  # files used to choose methylation panel


def build_local_copy_number_index() -> pd.DataFrame:
    """Index local copy-number segment files (not in legacy GDC mapping)."""
    from config import CANCER_TYPES, DATA_DIR

    rows = []
    cn_root = DATA_DIR / "copy_number"
    for project_id in CANCER_TYPES:
        cancer_dir = cn_root / project_id
        if not cancer_dir.exists():
            continue
        seen_patients = set()
        for seg_file in cancer_dir.iterdir():
            if not seg_file.is_file():
                continue
            _, barcode = parse_copy_number_seg(seg_file)
            if not barcode or barcode in seen_patients:
                continue
            seen_patients.add(barcode)
            rows.append({
                "case_submitter_id": barcode,
                "project_id": project_id,
                "local_path": str(seg_file),
                "data_type": "copy_number",
                "sample_type": "Primary Tumor",
                "has_local_file": True,
            })
    return pd.DataFrame(rows)


def primary_tumor_map(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    sub = df[
        (df["data_type"] == data_type)
        & (df["sample_type"] == "Primary Tumor")
        & (df["has_local_file"] == True)  # noqa: E712
    ].copy()
    return sub.drop_duplicates(subset=["case_submitter_id", "project_id"], keep="first")


def build_clinical_index():
    index = {}
    if not CLINICAL_CACHE.exists():
        return index
    for xml_file in CLINICAL_CACHE.glob("*.xml"):
        for part in xml_file.name.replace(".xml", "").split("."):
            if part.startswith("TCGA-"):
                patient = "-".join(part.split("-")[:3])
                index[patient] = xml_file
    return index


def parse_star_icgc_only(filepath: str | Path) -> dict[str, float]:
    wanted = set(ICGC_EXPRESSION_GENES)
    tpm = {}
    with open(filepath) as f:
        for line in f:
            if line.startswith("#") or line.startswith("N_"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 7:
                continue
            gene_name, gene_type = parts[1], parts[2]
            if gene_type != "protein_coding" or gene_name not in wanted:
                continue
            try:
                val = float(parts[6])
            except (ValueError, IndexError):
                continue
            if gene_name not in tpm or val > tpm[gene_name]:
                tpm[gene_name] = val
            if len(tpm) == len(wanted):
                break
    return tpm


def parse_methylation_subset(filepath: str | Path, probes: set[str]) -> dict[str, float]:
    out = {}
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2 or parts[0] not in probes:
                continue
            if parts[1] in ("NA", ""):
                continue
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                continue
            if len(out) == len(probes):
                break
    return out


def sample_probe_panel(meth_map: pd.DataFrame) -> list[str]:
    """Choose top variable CpGs from a stratified file sample."""
    probe_values = defaultdict(list)
    per_type = meth_map.groupby("project_id", group_keys=False).head(PROBE_SAMPLE_PER_TYPE)
    print(f"  Sampling {len(per_type)} methylation files for probe panel...")
    for _, row in per_type.iterrows():
        with open(row["local_path"]) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 2 or not parts[0].startswith("cg"):
                    continue
                if parts[1] in ("NA", ""):
                    continue
                try:
                    probe_values[parts[0]].append(float(parts[1]))
                except ValueError:
                    continue
    variances = {p: float(np.var(v)) for p, v in probe_values.items() if len(v) >= 5}
    top = sorted(variances, key=variances.get, reverse=True)[:TOP_METHYLATION_PROBES]
    return top


def main():
    print("=" * 60)
    print("Study 1 Step 2: Extract modality features")
    print("=" * 60)

    if not MAPPING_CSV.exists():
        print(f"ERROR: run step1 first ({MAPPING_CSV})")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mapping = pd.read_csv(MAPPING_CSV)
    clinical_index = build_clinical_index()
    print(f"Clinical XML index: {len(clinical_index)} patients")

    expr_map = primary_tumor_map(mapping, "expression")
    meth_map = primary_tumor_map(mapping, "methylation")
    mut_map = primary_tumor_map(mapping, "mutations")
    cn_map = primary_tumor_map(mapping, "copy_number")
    if cn_map.empty:
        cn_map = build_local_copy_number_index()
        print(f"  Using local CN index: {len(cn_map)} files")

    # --- Expression: ICGC proxy genes only ---
    print(f"\nParsing {len(expr_map)} expression files (ICGC genes only)...")
    icgc_rows = {}
    patient_cancer = {}
    for i, (_, row) in enumerate(expr_map.iterrows()):
        tpm = parse_star_icgc_only(row["local_path"])
        if not tpm:
            continue
        pid = row["case_submitter_id"]
        icgc_rows[pid] = icgc_expression_features(tpm)
        patient_cancer[pid] = row["project_id"]
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(expr_map)}")

    # --- Methylation: panel + subset read ---
    print(f"\nSelecting methylation probe panel...")
    top_probes = sample_probe_panel(meth_map)
    probe_set = set(top_probes)
    print(f"  Panel size: {len(top_probes)}")

    print(f"Parsing {len(meth_map)} methylation files (panel only)...")
    meth_rows = {}
    for i, (_, row) in enumerate(meth_map.iterrows()):
        probes = parse_methylation_subset(row["local_path"], probe_set)
        if not probes:
            continue
        pid = row["case_submitter_id"]
        meth_rows[pid] = {f"meth_{p}": probes.get(p, np.nan) for p in top_probes}
        patient_cancer.setdefault(pid, row["project_id"])
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(meth_map)}")

    # --- Mutations ---
    print(f"\nParsing {len(mut_map)} mutation files...")
    mut_rows = {}
    for i, (_, row) in enumerate(mut_map.iterrows()):
        feats, barcode = parse_maf_features(row["local_path"])
        pid = row["case_submitter_id"] if pd.notna(row["case_submitter_id"]) else barcode
        if not feats or not pid:
            continue
        selected = {
            "tmb_nonsyn": feats["tmb_nonsyn"],
            "tmb_total": feats["tmb_total"],
            "frameshift_count": feats["frameshift_count"],
        }
        for gene in MUTATION_GENES:
            selected[f"mut_{gene}"] = feats.get(f"mut_{gene}", 0.0)
        mut_rows[pid] = selected
        patient_cancer.setdefault(pid, row["project_id"])
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(mut_map)}")

    # --- Copy number + fragmentomics ---
    print(f"\nParsing {len(cn_map)} copy-number files...")
    cn_rows = {}
    frag_rows = {}
    for i, (_, row) in enumerate(cn_map.iterrows()):
        cn_feats, barcode = parse_copy_number_seg(row["local_path"])
        pid = row["case_submitter_id"] if pd.notna(row["case_submitter_id"]) else barcode
        if not cn_feats or not pid:
            continue
        cn_rows[pid] = cn_feats
        frag_rows[pid] = fragmentomics_from_segments(row["local_path"])
        patient_cancer.setdefault(pid, row["project_id"])
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(cn_map)}")

    # --- Clinical ---
    print("\nParsing clinical XML...")
    clin_rows = {pid: parse_clinical_xml(xml) for pid, xml in clinical_index.items()}

    pd.to_pickle(meth_rows, OUTPUT_DIR / "methylation_raw.pkl")
    pd.to_pickle(mut_rows, OUTPUT_DIR / "mutation_raw.pkl")
    pd.to_pickle(cn_rows, OUTPUT_DIR / "copy_number_raw.pkl")
    pd.to_pickle(frag_rows, OUTPUT_DIR / "fragmentomics_raw.pkl")
    pd.to_pickle(icgc_rows, OUTPUT_DIR / "icgc_expression_raw.pkl")
    pd.to_pickle(clin_rows, OUTPUT_DIR / "clinical_raw.pkl")
    pd.Series(patient_cancer).to_csv(OUTPUT_DIR / "patient_cancer_types.csv", header=["cancer_type"])

    with open(OUTPUT_DIR / "methylation_probe_panel.json", "w") as f:
        json.dump(top_probes, f, indent=2)

    summary = {
        "expression_patients": len(icgc_rows),
        "methylation_patients": len(meth_rows),
        "mutation_patients": len(mut_rows),
        "copy_number_patients": len(cn_rows),
        "clinical_patients": len(clin_rows),
        "icgc_proxy_patients": len(icgc_rows),
    }
    with open(OUTPUT_DIR / "step2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nModality coverage:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nSaved intermediates to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
