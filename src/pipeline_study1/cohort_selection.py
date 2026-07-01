"""Deterministic Study 1 cohort selection with modality completeness scoring."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import MAPPING_CSV, OUTPUT_DIR, RANDOM_STATE, TARGET_CLASS_COUNTS


CANONICAL_MANIFEST = OUTPUT_DIR / "canonical_patient_manifest.json"
CORE_DATA_TYPES = ("expression", "methylation", "mutation", "copy_number")


def load_completeness_scores(mapping_path: Path | None = None) -> pd.Series:
    """Higher score = more core modalities with local primary-tumor files."""
    path = mapping_path or MAPPING_CSV
    if not path.exists():
        return pd.Series(dtype=float)

    df = pd.read_csv(path)
    sub = df[
        (df["sample_type"] == "Primary Tumor")
        & (df["has_local_file"] == True)  # noqa: E712
        & (df["data_type"].isin(CORE_DATA_TYPES))
    ]
    counts = sub.groupby("case_submitter_id")["data_type"].nunique()
    return counts.astype(float)


def select_patients(
    pool: list[str],
    cancer_type: str,
    target: int,
    completeness: pd.Series | None = None,
    *,
    method: str = "deterministic",
    random_state: int = RANDOM_STATE,
) -> list[str]:
    """Select up to `target` patients from `pool` for one cancer type."""
    if not pool:
        return []

    if method == "random":
        n = min(target, len(pool))
        return pd.Series(pool).sample(n=n, random_state=random_state).tolist()

    scores = pd.Series({pid: completeness.get(pid, 0.0) if completeness is not None else 0.0 for pid in pool})
    ranked = (
        pd.DataFrame({"patient_id": pool, "score": scores.reindex(pool).fillna(0.0).values})
        .sort_values(["score", "patient_id"], ascending=[False, True])
    )
    return ranked.head(min(target, len(pool)))["patient_id"].tolist()


def select_cohort(
    labels: pd.Series,
    target_counts: dict[str, int] | None = None,
    completeness: pd.Series | None = None,
    *,
    method: str = "deterministic",
    manifest_path: Path | None = None,
) -> list[str]:
    """Select training cohort; honor pinned manifest when every class is satisfiable."""
    targets = target_counts or TARGET_CLASS_COUNTS
    manifest_file = manifest_path or CANONICAL_MANIFEST

    if manifest_file.exists():
        with open(manifest_file) as f:
            pinned = json.load(f)
        if isinstance(pinned, dict) and "patient_ids" in pinned:
            pinned_ids = pinned["patient_ids"]
        else:
            pinned_ids = pinned
        pinned_set = set(pinned_ids)
        ok = True
        for ct, n in targets.items():
            have = sum(1 for pid in pinned_ids if labels.get(pid) == ct)
            if have < n:
                ok = False
                break
        if ok:
            selected = []
            for ct, n in targets.items():
                class_ids = [pid for pid in pinned_ids if labels.get(pid) == ct][:n]
                selected.extend(class_ids)
            return selected

    selected: list[str] = []
    for ct, n in targets.items():
        pool = labels[labels == ct].index.tolist()
        selected.extend(select_patients(pool, ct, n, completeness, method=method))
    return selected


def write_canonical_manifest(patient_ids: list[str], labels: pd.Series, source: str) -> None:
    """Persist the cohort used for reproducibility."""
    class_counts = pd.Series([labels[pid] for pid in patient_ids]).value_counts().to_dict()
    payload = {
        "patient_ids": patient_ids,
        "n_samples": len(patient_ids),
        "class_counts": class_counts,
        "target_class_counts": TARGET_CLASS_COUNTS,
        "selection_method": source,
        "note": (
            "No legacy 158-barcode list was found in the repository. "
            "This manifest pins the reproduced Study 1 training cohort."
        ),
    }
    with open(CANONICAL_MANIFEST, "w") as f:
        json.dump(payload, f, indent=2)
