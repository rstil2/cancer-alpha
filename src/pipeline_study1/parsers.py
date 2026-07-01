"""File parsers for Study 1 multi-modal feature extraction."""

from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from config import ICGC_EXPRESSION_GENES, MUTATION_GENES

NONSYNONYMOUS = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
    "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins", "Splice_Site",
}


def patient_id_from_barcode(barcode: str) -> str | None:
    if not barcode or not barcode.startswith("TCGA-"):
        return None
    parts = barcode.split("-")
    if len(parts) >= 3:
        return "-".join(parts[:3])
    return None


def parse_star_tpm(filepath: str | Path) -> dict[str, float]:
    gene_tpm: dict[str, float] = {}
    with open(filepath) as f:
        for line in f:
            if line.startswith("#") or line.startswith("N_"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 7:
                continue
            gene_name, gene_type = parts[1], parts[2]
            if gene_type != "protein_coding":
                continue
            try:
                tpm = float(parts[6])
            except (ValueError, IndexError):
                continue
            if gene_name not in gene_tpm or tpm > gene_tpm[gene_name]:
                gene_tpm[gene_name] = tpm
    return gene_tpm


def parse_methylation_betas(filepath: str | Path) -> dict[str, float]:
    probes: dict[str, float] = {}
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2 or not parts[0].startswith("cg"):
                continue
            if parts[1] in ("NA", ""):
                continue
            try:
                probes[parts[0]] = float(parts[1])
            except ValueError:
                continue
    return probes


def parse_maf_features(filepath: str | Path) -> tuple[dict[str, float], str | None]:
    open_fn = gzip.open if str(filepath).endswith(".gz") else open
    mode = "rt" if str(filepath).endswith(".gz") else "r"
    mutations = []
    barcode = None
    with open_fn(filepath, mode) as f:
        header = None
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if header is None:
                header = parts
                continue
            if len(parts) < len(header):
                continue
            row = dict(zip(header, parts))
            if barcode is None:
                bc = row.get("Tumor_Sample_Barcode", "")
                barcode = patient_id_from_barcode(bc)
            mutations.append({
                "gene": row.get("Hugo_Symbol", ""),
                "variant_class": row.get("Variant_Classification", ""),
            })
    if not mutations:
        return {}, barcode

    nonsyn = [m for m in mutations if m["variant_class"] in NONSYNONYMOUS]
    mutated = {m["gene"] for m in nonsyn}
    features: dict[str, float] = {
        "tmb_nonsyn": float(len(nonsyn)),
        "tmb_total": float(len(mutations)),
        "frameshift_count": float(sum(
            1 for m in nonsyn if "Frame_Shift" in m["variant_class"]
        )),
    }
    for gene in MUTATION_GENES:
        features[f"mut_{gene}"] = 1.0 if gene in mutated else 0.0
    return features, barcode


def _segment_arrays(segments: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.array([max(e - s, 1) for _, s, e, _ in segments], dtype=float)
    means = np.array([m for *_, m in segments], dtype=float)
    return lengths, means


def copy_number_features_from_tuples(segments: list[tuple]) -> dict[str, float]:
    """CN features from (chrom, start, end, mean) segment tuples."""
    if not segments:
        return {}
    lengths, means = _segment_arrays(segments)
    weights = lengths / lengths.sum()
    features: dict[str, float] = {
        "cn_n_segments": float(len(segments)),
        "cn_mean_segment_mb": float(np.mean(lengths) / 1e6),
        "cn_std_segment_mb": float(np.std(lengths) / 1e6),
        "cn_weighted_mean": float(np.average(means, weights=weights)),
        "cn_weighted_std": float(np.sqrt(np.average((means - np.average(means, weights=weights)) ** 2, weights=weights))),
        "cn_frac_amp": float(np.mean(means > 0.2)),
        "cn_frac_del": float(np.mean(means < -0.2)),
        "cn_max_amp": float(np.max(means)),
        "cn_max_del": float(np.min(means)),
        "cn_genome_altered": float(np.sum(weights[np.abs(means) > 0.2])),
    }
    chrom_order = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
    for i, chrom in enumerate(chrom_order[:10]):
        idx = [j for j, (c, _, _, _) in enumerate(segments) if c == chrom]
        if not idx:
            features[f"cn_{chrom}"] = 0.0
            continue
        c_lengths = lengths[idx]
        c_means = means[idx]
        features[f"cn_{chrom}"] = float(np.average(c_means, weights=c_lengths / c_lengths.sum()))
    return features


def fragmentomics_from_segment_tuples(segments: list[tuple]) -> dict[str, float]:
    """Fragmentomics proxies from (chrom, start, end, mean) tuples."""
    if not segments:
        return {}
    lengths = np.array([max(e - s, 1) for _, s, e, _ in segments], dtype=float)
    means = np.array([m for *_, m in segments], dtype=float)
    short = lengths[lengths < np.percentile(lengths, 25)]
    long = lengths[lengths > np.percentile(lengths, 75)]
    return {
        "frag_median_kb": float(np.median(lengths) / 1000),
        "frag_mean_kb": float(np.mean(lengths) / 1000),
        "frag_std_kb": float(np.std(lengths) / 1000),
        "frag_short_median_kb": float(np.median(short) / 1000) if len(short) else 0.0,
        "frag_long_median_kb": float(np.median(long) / 1000) if len(long) else 0.0,
        "frag_short_long_ratio": float(np.median(short) / max(np.median(long), 1)),
        "frag_amp_burden": float(np.mean(means > 0.2)),
        "frag_del_burden": float(np.mean(means < -0.2)),
        "frag_abs_mean": float(np.mean(np.abs(means))),
        "frag_abs_std": float(np.std(np.abs(means))),
        "frag_skew": float(float(np.mean(((means - means.mean()) / (means.std() + 1e-6)) ** 3))),
        "frag_high_amp_frac": float(np.mean(means > 0.5)),
        "frag_high_del_frac": float(np.mean(means < -0.5)),
        "frag_heterogeneity": float(np.std(means)),
        "frag_length_entropy": float(-np.sum((lengths / lengths.sum()) * np.log(lengths / lengths.sum() + 1e-12))),
    }


def parse_copy_number_seg(filepath: str | Path) -> tuple[dict[str, float], str | None]:
    segments = []
    barcode = None
    with open(filepath) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < len(header):
                continue
            row = dict(zip(header, parts))
            aliquot = row.get("GDC_Aliquot_ID") or row.get("Sample") or ""
            if barcode is None and aliquot.startswith("TCGA-"):
                barcode = patient_id_from_barcode(aliquot)
            try:
                start = int(float(row.get("Start", 0)))
                end = int(float(row.get("End", 0)))
                mean = float(row.get("Segment_Mean", 0))
            except (TypeError, ValueError):
                continue
            chrom = row.get("Chromosome", "")
            segments.append((chrom, start, end, mean))

    if not segments:
        return {}, barcode

    features = copy_number_features_from_tuples(segments)
    return features, barcode


def fragmentomics_from_segments(filepath: str | Path) -> dict[str, float]:
    """Fragmentomics proxies derived from copy-number segment profiles."""
    segments = []
    with open(filepath) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < len(header):
                continue
            row = dict(zip(header, parts))
            try:
                start = int(float(row.get("Start", 0)))
                end = int(float(row.get("End", 0)))
                mean = float(row.get("Segment_Mean", 0))
            except (TypeError, ValueError):
                continue
            segments.append((end - start, mean))

    if not segments:
        return {f"frag_{i}": 0.0 for i in range(15)}

    lengths = np.array([max(l, 1) for l, _ in segments], dtype=float)
    means = np.array([m for _, m in segments], dtype=float)
    short = lengths[lengths < np.percentile(lengths, 25)]
    long = lengths[lengths > np.percentile(lengths, 75)]

    return {
        "frag_median_kb": float(np.median(lengths) / 1000),
        "frag_mean_kb": float(np.mean(lengths) / 1000),
        "frag_std_kb": float(np.std(lengths) / 1000),
        "frag_short_median_kb": float(np.median(short) / 1000) if len(short) else 0.0,
        "frag_long_median_kb": float(np.median(long) / 1000) if len(long) else 0.0,
        "frag_short_long_ratio": float(np.median(short) / max(np.median(long), 1)),
        "frag_amp_burden": float(np.mean(means > 0.2)),
        "frag_del_burden": float(np.mean(means < -0.2)),
        "frag_abs_mean": float(np.mean(np.abs(means))),
        "frag_abs_std": float(np.std(np.abs(means))),
        "frag_skew": float(float(np.mean(((means - means.mean()) / (means.std() + 1e-6)) ** 3))),
        "frag_high_amp_frac": float(np.mean(means > 0.5)),
        "frag_high_del_frac": float(np.mean(means < -0.5)),
        "frag_heterogeneity": float(np.std(means)),
        "frag_length_entropy": float(-np.sum((lengths / lengths.sum()) * np.log(lengths / lengths.sum() + 1e-12))),
    }


def icgc_expression_features(gene_tpm: dict[str, float]) -> dict[str, float]:
    return {f"icgc_{g}": float(np.log2(gene_tpm.get(g, 0.0) + 1.0)) for g in ICGC_EXPRESSION_GENES}


def parse_clinical_xml(filepath: str | Path) -> dict[str, float]:
    features = {
        "clin_age": np.nan,
        "clin_gender": np.nan,
        "clin_vital": np.nan,
        "clin_stage": np.nan,
        "clin_grade": np.nan,
        "clin_days_to_death": np.nan,
        "clin_days_to_last_followup": np.nan,
        "clin_tumor_status": np.nan,
        "clin_neoplasm": np.nan,
        "clin_lymph_nodes": np.nan,
    }
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except ET.ParseError:
        return features

    def walk(elem):
        tag = elem.tag.split("}")[-1].lower()
        text = (elem.text or "").strip()
        if tag in ("age_at_diagnosis", "age_at_initial_pathologic_diagnosis") and text:
            try:
                features["clin_age"] = float(text) / 100.0
            except ValueError:
                pass
        elif tag == "gender" and text:
            features["clin_gender"] = 1.0 if text.upper() == "MALE" else 0.0
        elif tag == "vital_status" and text:
            features["clin_vital"] = 1.0 if text.lower() == "dead" else 0.0
        elif tag in ("tumor_stage", "pathologic_stage", "clinical_stage") and text:
            features["clin_stage"] = _encode_stage(text) / 4.0
        elif tag in ("histological_grade", "neoplasm_histologic_grade") and text:
            features["clin_grade"] = _encode_grade(text) / 4.0
        elif tag == "days_to_death" and text:
            try:
                features["clin_days_to_death"] = float(text) / 10000.0
            except ValueError:
                pass
        elif tag == "days_to_last_followup" and text:
            try:
                features["clin_days_to_last_followup"] = float(text) / 10000.0
            except ValueError:
                pass
        elif tag == "person_neoplasm_cancer_status" and text:
            features["clin_tumor_status"] = 1.0 if "tumor" in text.lower() else 0.0
        elif tag == "lymph_node_examined_count" and text:
            try:
                features["clin_lymph_nodes"] = float(text) / 50.0
            except ValueError:
                pass
        elif tag in ("new_tumor_event_after_initial_treatment",) and text:
            features["clin_neoplasm"] = 1.0 if text.upper() == "YES" else 0.0
        for child in elem:
            walk(child)

    walk(root)
    return features


def _encode_stage(stage_str: str) -> float:
    s = stage_str.upper()
    if "IV" in s or "4" in s:
        return 4.0
    if "III" in s or "3" in s:
        return 3.0
    if "II" in s or "2" in s:
        return 2.0
    if "I" in s or "1" in s:
        return 1.0
    return 2.0


def _encode_grade(grade_str: str) -> float:
    s = str(grade_str).upper()
    for val, token in [(4, "4"), (4, "IV"), (3, "3"), (3, "III"), (2, "2"), (2, "II"), (1, "1"), (1, "I")]:
        if token in s:
            return float(val)
    return 2.0
