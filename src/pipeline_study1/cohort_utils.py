"""Shared feature integration for Study 1 cohorts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import MUTATION_GENES, OUTPUT_DIR, TARGET_CLASS_COUNTS, TOTAL_FEATURES


def ordered_feature_names(meth_probes: list[str]) -> list[str]:
    names = [f"meth_{p}" for p in meth_probes]
    names += ["tmb_nonsyn", "tmb_total", "frameshift_count"]
    names += [f"mut_{g}" for g in MUTATION_GENES]
    names += [
        "cn_n_segments", "cn_mean_segment_mb", "cn_std_segment_mb",
        "cn_weighted_mean", "cn_weighted_std", "cn_frac_amp", "cn_frac_del",
        "cn_max_amp", "cn_max_del", "cn_genome_altered",
        "cn_chr1", "cn_chr2", "cn_chr3", "cn_chr4", "cn_chr5",
        "cn_chr6", "cn_chr7", "cn_chr8", "cn_chr9", "cn_chr10",
    ]
    names += [
        "frag_median_kb", "frag_mean_kb", "frag_std_kb", "frag_short_median_kb",
        "frag_long_median_kb", "frag_short_long_ratio", "frag_amp_burden",
        "frag_del_burden", "frag_abs_mean", "frag_abs_std", "frag_skew",
        "frag_high_amp_frac", "frag_high_del_frac",
        "frag_heterogeneity", "frag_length_entropy",
    ]
    names += [
        "clin_age", "clin_gender", "clin_vital", "clin_stage", "clin_grade",
        "clin_days_to_death", "clin_days_to_last_followup", "clin_tumor_status",
        "clin_neoplasm", "clin_lymph_nodes",
    ]
    names += [f"icgc_{g}" for g in [
        "ESR1", "ERBB2", "EGFR", "KRAS", "BRAF", "PIK3CA", "TP53", "BRCA1",
        "BRCA2", "MET", "ALK", "RET", "FGFR2", "CDK4", "CCND1", "MDM2",
        "PTEN", "STK11", "NRAS", "MYC",
    ]]
    if len(names) != TOTAL_FEATURES:
        raise ValueError(f"Expected {TOTAL_FEATURES} features, got {len(names)}")
    return names


def load_modality_dicts() -> tuple[dict, dict, dict, dict, dict, dict, pd.Series]:
    meth = pd.read_pickle(OUTPUT_DIR / "methylation_raw.pkl")
    mut = pd.read_pickle(OUTPUT_DIR / "mutation_raw.pkl")
    cn = pd.read_pickle(OUTPUT_DIR / "copy_number_raw.pkl")
    frag = pd.read_pickle(OUTPUT_DIR / "fragmentomics_raw.pkl")
    icgc = pd.read_pickle(OUTPUT_DIR / "icgc_expression_raw.pkl")
    clin = pd.read_pickle(OUTPUT_DIR / "clinical_raw.pkl") if (OUTPUT_DIR / "clinical_raw.pkl").exists() else {}
    cancer_types = pd.read_csv(OUTPUT_DIR / "patient_cancer_types.csv", index_col=0)["cancer_type"]
    return meth, mut, cn, frag, icgc, clin, cancer_types


def common_patients(meth, mut, cn, frag, icgc) -> set[str]:
    return set(meth) & set(mut) & set(cn) & set(frag) & set(icgc)


def build_feature_matrix(
    patient_ids: list[str],
    meth_probes: list[str],
    meth, mut, cn, frag, icgc, clin, cancer_types,
    allowed_types: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    feature_names = ordered_feature_names(meth_probes)
    rows, labels, kept = [], [], []
    for pid in patient_ids:
        ct = cancer_types.get(pid)
        if ct is None or (allowed_types and ct not in allowed_types):
            continue
        row = {}
        row.update(meth.get(pid, {}))
        row.update(mut.get(pid, {}))
        row.update(cn.get(pid, {}))
        row.update(frag.get(pid, {}))
        row.update(clin.get(pid, {k: np.nan for k in feature_names if k.startswith("clin_")}))
        row.update(icgc.get(pid, {}))
        rows.append([row.get(k, np.nan) for k in feature_names])
        labels.append(ct)
        kept.append(pid)
    X = pd.DataFrame(rows, index=kept, columns=feature_names, dtype=float)
    y = pd.Series(labels, index=kept, name="cancer_type")
    return X, y


def load_meth_probes() -> list[str]:
    with open(OUTPUT_DIR / "methylation_probe_panel.json") as f:
        return json.load(f)
