"""Fetch real ICGC donor data from the public UCSC Xena ICGC hub."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

try:
    import xenaPython as xp
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install xenaPython: pip install xenaPython") from exc

from config import ICGC_EXPRESSION_GENES, MUTATION_GENES

ICGC_HOST = "https://icgc.xenahubs.net"
PHENO_DS = "donor/donor.all_projects.phenotype"
MUT_DS = "donor/SNV.donor.codingMutation-allProjects"
MUT_NON_US_DS = "donor/SNV.donor.allMutation-nonUSprojects"
EXP_DS = "donor/exp_seq.all_projects.donor.USonly.xena.tsv"
CN_DS = "donor/copy_number_somatic_mutation.all_projects.donor"

# US ICGC projects that overlap TCGA — exclude for external validation
EXCLUDED_PROJECTS = {
    "BRCA-US", "LUAD-US", "LUSC-US", "COAD-US", "READ-US", "PRAD-US",
    "ALL-US",
}

# Manuscript ICGC panel (4 types, n=76)
ICGC_TARGET_COUNTS = {
    "TCGA-BRCA": 28,
    "TCGA-LUAD": 22,
    "TCGA-COAD": 16,
    "TCGA-PRAD": 10,
}

# _primary_site code indices (from field_codes)
SITE_BREAST = 1
SITE_COLORECTAL = 2
SITE_PROSTATE = 3
SITE_LUNG = 11

# _primary_disease code indices for finer LUAD vs LUSC
DISEASE_LUAD = {17, 44}
DISEASE_COAD = {2, 41, 66}
DISEASE_BREAST = {1, 6, 11, 46, 50}
DISEASE_PROSTATE = {3, 21, 63}


def _parse_codes(host: str, dataset: str, fields: list[str]) -> dict[str, list[str]]:
    raw = xp.field_codes(host, dataset, fields)
    return {item["name"]: item["code"].split("\t") for item in raw}


def _map_donor_label(site: int | None, disease: int | None) -> str | None:
    if disease in DISEASE_BREAST or site == SITE_BREAST:
        return "TCGA-BRCA"
    if disease in DISEASE_PROSTATE or site == SITE_PROSTATE:
        return "TCGA-PRAD"
    if disease in DISEASE_COAD or site == SITE_COLORECTAL:
        return "TCGA-COAD"
    if disease in DISEASE_LUAD or site == SITE_LUNG:
        return "TCGA-LUAD"
    return None


def fetch_donor_table(host: str = ICGC_HOST) -> list[dict]:
    """Return donor metadata with decoded project and cancer labels."""
    codes = _parse_codes(host, PHENO_DS, ["project_code", "_primary_site", "_primary_disease"])
    project_names = codes["project_code"]
    samples = xp.dataset_samples(host, PHENO_DS, None)
    fields = ["project_code", "_primary_site", "_primary_disease", "donor_sex", "donor_age_at_diagnosis"]
    cols = xp.dataset_fetch(host, PHENO_DS, samples, fields)
    rows = []
    for i, donor_id in enumerate(samples):
        proj_idx = cols[0][i]
        site_idx = cols[1][i]
        dis_idx = cols[2][i]
        project = project_names[proj_idx] if isinstance(proj_idx, int) and proj_idx < len(project_names) else None
        if project in EXCLUDED_PROJECTS:
            continue
        label = _map_donor_label(site_idx, dis_idx)
        if label is None:
            continue
        rows.append({
            "donor_id": donor_id,
            "project_code": project,
            "tcga_label": label,
            "donor_sex": cols[3][i],
            "donor_age_at_diagnosis": cols[4][i],
        })
    return rows


def _available_sets(host: str) -> dict[str, set[str]]:
    return {
        "mutation": set(xp.dataset_samples(host, MUT_DS, None)),
        "mutation_non_us": set(xp.dataset_samples(host, MUT_NON_US_DS, None)),
        "expression": set(xp.dataset_samples(host, EXP_DS, None)),
        "copy_number": set(xp.dataset_samples(host, CN_DS, None)),
    }


def select_icgc_donors(
    donors: list[dict],
    available: dict[str, set[str]],
    targets: dict[str, int] | None = None,
) -> list[dict]:
    """Pick donors with mutation data, preferring expression+CN when available."""
    targets = targets or ICGC_TARGET_COUNTS
    mut_pool = available["mutation"] | available["mutation_non_us"]
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in donors:
        if row["donor_id"] not in mut_pool:
            continue
        score = 0
        if row["donor_id"] in available["expression"]:
            score += 2
        if row["donor_id"] in available["copy_number"]:
            score += 1
        row = {**row, "modality_score": score}
        by_label[row["tcga_label"]].append(row)

    selected: list[dict] = []
    for label, target in targets.items():
        pool = sorted(
            by_label.get(label, []),
            key=lambda r: (-r["modality_score"], r["donor_id"]),
        )
        selected.extend(pool[: min(target, len(pool))])
    return selected


def fetch_expression(host: str, donor_ids: list[str], genes: list[str] | None = None) -> dict[str, dict[str, float]]:
    genes = genes or ICGC_EXPRESSION_GENES
    avail = set(xp.dataset_samples(host, EXP_DS, None))
    ids = [d for d in donor_ids if d in avail]
    out: dict[str, dict[str, float]] = {}
    if not ids:
        return out
    batch = 50
    for i in range(0, len(ids), batch):
        chunk = ids[i : i + batch]
        result = xp.dataset_probe_values(host, EXP_DS, chunk, genes)
        # result: [positions, [gene_values_per_sample]]
        if not result or result[1] is None:
            continue
        for j, donor in enumerate(chunk):
            tpm = {}
            for g_idx, gene in enumerate(genes):
                try:
                    val = result[1][g_idx][j]
                except (IndexError, TypeError):
                    val = 0.0
                tpm[gene] = float(val or 0.0)
            out[donor] = {f"icgc_{g}": float(np.log2(tpm[g] + 1.0)) for g in genes}
    return out


def fetch_mutations(host: str, donor_ids: list[str]) -> dict[str, dict[str, float]]:
    """Build mutation summary + per-gene indicators from sparse mutation vectors."""
    out: dict[str, dict[str, float]] = {}
    genes = list(MUTATION_GENES)
    for donor in donor_ids:
        data = xp.sparse_data(host, MUT_DS, [donor], genes)
        rows = data.get("rows", {}) if isinstance(data, dict) else {}
        mut_genes = rows.get("genes", []) or []
        effects = rows.get("effect", []) or []
        flat_genes: list[str] = []
        for item in mut_genes:
            if isinstance(item, list):
                flat_genes.extend(str(g) for g in item)
            elif item:
                flat_genes.append(str(item))
        gene_hits = {g for g in flat_genes if g in MUTATION_GENES}
        nonsyn = 0
        frameshift = 0
        for eff in effects:
            eff_s = str(eff).lower()
            if eff_s and "synonymous" not in eff_s:
                nonsyn += 1
            if "frame" in eff_s:
                frameshift += 1
        total = len(flat_genes)
        feats = {
            "tmb_nonsyn": float(nonsyn),
            "tmb_total": float(total),
            "frameshift_count": float(frameshift),
        }
        feats.update({f"mut_{g}": 1.0 if g in gene_hits else 0.0 for g in MUTATION_GENES})
        out[donor] = feats
    return out


def fetch_copy_number_features(host: str, donor_ids: list[str]) -> dict[str, dict[str, float]]:
    """CN + fragmentomics proxies from segmented copy-number (chr1–10)."""
    from parsers import copy_number_features_from_tuples, fragmentomics_from_segment_tuples

    avail = set(xp.dataset_samples(host, CN_DS, None))
    out: dict[str, dict[str, float]] = {}
    for donor in donor_ids:
        if donor not in avail:
            continue
        segments = []
        for chrom in range(1, 11):
            chrom_name = f"chr{chrom}"
            try:
                seg = xp.segmented_data_range(host, CN_DS, [donor], chrom_name, 1, 250_000_000)
            except Exception:
                continue
            rows = seg.get("rows", {}) if isinstance(seg, dict) else {}
            chrom_raw = rows.get("chrom", rows.get("chromosome", [])) or []
            starts = rows.get("start", []) or []
            ends = rows.get("end", []) or []
            values = rows.get("value", []) or []
            for chrom, s, e, v in zip(chrom_raw, starts, ends, values):
                try:
                    chrom_str = str(chrom)
                    if not chrom_str.startswith("chr"):
                        chrom_str = f"chr{chrom_str}"
                    segments.append((chrom_str, int(s), int(e), float(v)))
                except (TypeError, ValueError):
                    continue
        if not segments:
            continue
        parsed = {**copy_number_features_from_tuples(segments), **fragmentomics_from_segment_tuples(segments)}
        out[donor] = parsed
    return out


def clinical_features_from_donor(row: dict) -> dict[str, float]:
    sex = row.get("donor_sex")
    age = row.get("donor_age_at_diagnosis")
    return {
        "clin_age": float(age) / 100.0 if isinstance(age, (int, float)) and age == age else np.nan,
        "clin_gender": 1.0 if sex == 1 else (0.0 if sex == 0 else np.nan),
        "clin_vital": np.nan,
        "clin_stage": np.nan,
        "clin_grade": np.nan,
        "clin_days_to_death": np.nan,
        "clin_days_to_last_followup": np.nan,
        "clin_tumor_status": np.nan,
        "clin_neoplasm": np.nan,
        "clin_lymph_nodes": np.nan,
    }
