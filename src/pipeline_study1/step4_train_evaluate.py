#!/usr/bin/env python3
"""Study 1 Step 4: Train LightGBM+SMOTE and evaluate (5-fold CV, multi-seed)."""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler

import lightgbm as lgb

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import N_CV_FOLDS, N_INDEPENDENT_RUNS, OUTPUT_DIR, RANDOM_STATE, SMOTE_K_NEIGHBORS


def make_pipeline(le: LabelEncoder):
    return ImbPipeline([
        ("scaler", RobustScaler()),
        ("smote", SMOTE(
            random_state=RANDOM_STATE,
            k_neighbors=SMOTE_K_NEIGHBORS,
            sampling_strategy="auto",
        )),
        ("clf", lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(le.classes_),
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbosity=-1,
        )),
    ])


def cv_balanced_accuracy(X, y_enc, seed: int) -> float:
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=seed)
    scores = []
    le = LabelEncoder()
    le.fit(range(len(np.unique(y_enc))))
    for train_idx, val_idx in skf.split(X, y_enc):
        pipe = make_pipeline(le)
        pipe.named_steps["smote"].set_params(random_state=seed)
        pipe.named_steps["clf"].set_params(random_state=seed)
        pipe.fit(X[train_idx], y_enc[train_idx])
        pred = pipe.predict(X[val_idx])
        scores.append(balanced_accuracy_score(y_enc[val_idx], pred))
    return float(np.mean(scores))


def main():
    print("=" * 60)
    print("Study 1 Step 4: LightGBM + SMOTE evaluation")
    print("=" * 60)

    feat_path = OUTPUT_DIR / "features_110.pkl"
    label_path = OUTPUT_DIR / "labels.csv"
    if not feat_path.exists() or not label_path.exists():
        print("ERROR: run step3 first.")
        sys.exit(1)

    X_df = pd.read_pickle(feat_path)
    y_series = pd.read_csv(label_path, index_col=0).squeeze()
    if isinstance(y_series, pd.DataFrame):
        y_series = y_series.iloc[:, 0]

    le = LabelEncoder()
    y_enc = le.fit_transform(y_series.values)
    X = X_df.values.astype(float)

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(le.classes_)} classes")

    run_scores = []
    for run in range(N_INDEPENDENT_RUNS):
        seed = RANDOM_STATE + run
        score = cv_balanced_accuracy(X, y_enc, seed)
        run_scores.append(score)
        print(f"  Run {run + 1}/{N_INDEPENDENT_RUNS} (seed={seed}): {score:.4f}")

    cv_mean = float(np.mean(run_scores))
    cv_std = float(np.std(run_scores, ddof=1)) if len(run_scores) > 1 else 0.0

    # Final model on full cohort for serialization
    final_pipe = make_pipeline(le)
    final_pipe.fit(X, y_enc)
    y_pred = final_pipe.predict(X)
    train_bal = float(balanced_accuracy_score(y_enc, y_pred))

    import joblib
    joblib.dump(final_pipe, OUTPUT_DIR / "lightgbm_smote_study1.pkl")
    joblib.dump(le, OUTPUT_DIR / "label_encoder.pkl")

    report = classification_report(
        y_enc, y_pred, target_names=le.classes_, output_dict=True, zero_division=0,
    )

    results = {
        "model": "LightGBM+SMOTE",
        "cv_runs": run_scores,
        "cv_mean_balanced_accuracy": cv_mean,
        "cv_std_balanced_accuracy": cv_std,
        "cv_mean_pct": round(cv_mean * 100, 1),
        "cv_std_pct": round(cv_std * 100, 1),
        "train_balanced_accuracy": train_bal,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "smote_k_neighbors": SMOTE_K_NEIGHBORS,
        "n_folds": N_CV_FOLDS,
        "n_independent_runs": N_INDEPENDENT_RUNS,
        "classification_report_train": report,
        "class_names": le.classes_.tolist(),
    }

    with open(OUTPUT_DIR / "model_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 40}")
    print(f"LightGBM+SMOTE: {cv_mean:.1%} ± {cv_std:.1%} balanced accuracy")
    print(f"  ({N_INDEPENDENT_RUNS} runs × {N_CV_FOLDS}-fold CV)")
    print(f"Saved model -> {OUTPUT_DIR / 'lightgbm_smote_study1.pkl'}")
    print(f"Saved results -> {OUTPUT_DIR / 'model_results.json'}")


if __name__ == "__main__":
    main()
