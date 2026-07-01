#!/usr/bin/env python3
"""Study 1 Step 5: External validation without retraining (held-out TCGA + optional ICGC file)."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUT_DIR, PROJECT_ROOT

ICGC_OPTIONAL = PROJECT_ROOT / "data" / "icgc_argo" / "features_110.pkl"
ICGC_LABELS = PROJECT_ROOT / "data" / "icgc_argo" / "labels.csv"


def score_cohort(name: str, X: pd.DataFrame, y: pd.Series, pipe, le) -> dict:
    # Align to model classes
    mask = y.isin(le.classes_)
    X = X.loc[mask]
    y = y.loc[mask]
    if len(X) == 0:
        return {"cohort": name, "n_samples": 0, "error": "no overlapping classes"}

    y_enc = le.transform(y.values)
    pred = pipe.predict(X.values.astype(float))
    bal = float(balanced_accuracy_score(y_enc, pred))
    report = classification_report(
        y_enc, pred, labels=range(len(le.classes_)),
        target_names=le.classes_, output_dict=True, zero_division=0,
    )
    per_class = {
        le.classes_[i]: {
            "precision": report[le.classes_[i]]["precision"],
            "recall": report[le.classes_[i]]["recall"],
            "support": report[le.classes_[i]]["support"],
        }
        for i in range(len(le.classes_))
        if le.classes_[i] in report
    }
    return {
        "cohort": name,
        "n_samples": int(len(X)),
        "balanced_accuracy": bal,
        "balanced_accuracy_pct": round(bal * 100, 1),
        "per_class": per_class,
    }


def main():
    print("=" * 60)
    print("Study 1 Step 5: External validation (no retraining)")
    print("=" * 60)

    model_path = OUTPUT_DIR / "lightgbm_smote_study1.pkl"
    le_path = OUTPUT_DIR / "label_encoder.pkl"
    ext_X_path = OUTPUT_DIR / "external_features_110.pkl"
    ext_y_path = OUTPUT_DIR / "external_labels.csv"

    for p in (model_path, le_path, ext_X_path, ext_y_path):
        if not p.exists():
            print(f"ERROR: missing {p}. Run steps 3b and 4 first.")
            sys.exit(1)

    pipe = joblib.load(model_path)
    le = joblib.load(le_path)
    X_ext = pd.read_pickle(ext_X_path)
    y_ext = pd.read_csv(ext_y_path, index_col=0).squeeze()
    if isinstance(y_ext, pd.DataFrame):
        y_ext = y_ext.iloc[:, 0]

    results = {
        "model": "LightGBM+SMOTE (frozen, trained on n=158)",
        "validation_runs": [],
    }

    tcga_result = score_cohort("tcga_held_out_4type", X_ext, y_ext, pipe, le)
    results["validation_runs"].append(tcga_result)
    print(f"\nTCGA held-out (4-type, n={tcga_result.get('n_samples', 0)}): "
          f"{tcga_result.get('balanced_accuracy_pct', 'N/A')}% balanced accuracy")

    if ICGC_OPTIONAL.exists() and ICGC_LABELS.exists():
        X_icgc = pd.read_pickle(ICGC_OPTIONAL)
        y_icgc = pd.read_csv(ICGC_LABELS, index_col=0).squeeze()
        icgc_result = score_cohort("icgc_argo_file", X_icgc, y_icgc, pipe, le)
        results["validation_runs"].append(icgc_result)
        print(f"ICGC ARGO file (n={icgc_result.get('n_samples', 0)}): "
              f"{icgc_result.get('balanced_accuracy_pct', 'N/A')}% balanced accuracy")
    else:
        results["icgc_argo_note"] = (
            f"No ICGC feature file at {ICGC_OPTIONAL}. "
            "TCGA held-out external validation is reported instead."
        )
        print(f"\nICGC file not found at {ICGC_OPTIONAL} (optional).")

    with open(OUTPUT_DIR / "external_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved -> {OUTPUT_DIR / 'external_validation_results.json'}")


if __name__ == "__main__":
    main()
