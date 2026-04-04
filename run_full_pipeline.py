#!/usr/bin/env python3
"""
Oncura — Full Reproducible Pipeline
====================================

End-to-end pipeline that reproduces the manuscript results:
  1. Load pre-extracted TCGA features (expression, methylation, mutations)
  2. Integrate modalities by patient ID (three-way intersection)
  3. Balance classes by down-sampling to the smallest class
  4. Train/test split (80/20, stratified)
  5. Train LightGBM (Bayesian optimization via Optuna), XGBoost, RF, LR
  6. Evaluate on held-out test set (balanced accuracy, per-class metrics)
  7. Stratified 5-fold cross-validation
  8. SHAP feature importance analysis
  9. Save all results and the best model

Prerequisites:
  pip install numpy pandas scikit-learn lightgbm xgboost optuna

Usage:
  python run_full_pipeline.py                # Full pipeline
  python run_full_pipeline.py --skip-optuna  # Skip Bayesian optimization (faster)

Data files required (in data/real_model_results/):
  - expression_features.pkl   (2,000 gene expression features per patient)
  - methylation_features.pkl  (2,000 CpG probe features per patient)
  - mutation_features.pkl     (63 mutation-derived features per patient)
  - expression_metadata.csv   (patient_id → cancer_type mapping)

These feature files were extracted from TCGA GDC portal data:
  - Gene expression: STAR workflow TPM values, log2(TPM+1), top 2,000 by variance
  - DNA methylation: SeSAMe beta values, top 2,000 CpG probes by variance
  - Somatic mutations: 63 features (TMB, 51 driver gene indicators, 9 variant types)

Author: Dr. R. Craig Stillwell
Patent: Provisional Application No. 63/847,316
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "real_model_results"
OUTPUT_DIR = PROJECT_ROOT / "pipeline_output"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_CV_FOLDS = 5


# ── Step 1: Load Features ────────────────────────────────────────────────────

def load_features():
    """Load pre-extracted TCGA feature matrices for all three modalities."""
    print("=" * 70)
    print("STEP 1: Loading TCGA Feature Matrices")
    print("=" * 70)

    expr_df = pd.read_pickle(DATA_DIR / "expression_features.pkl")
    meth_df = pd.read_pickle(DATA_DIR / "methylation_features.pkl")
    mut_df = pd.read_pickle(DATA_DIR / "mutation_features.pkl")
    meta_df = pd.read_csv(DATA_DIR / "expression_metadata.csv")

    print(f"  Gene expression:  {expr_df.shape[0]:,} patients × {expr_df.shape[1]:,} features")
    print(f"  DNA methylation:  {meth_df.shape[0]:,} patients × {meth_df.shape[1]:,} features")
    print(f"  Somatic mutations: {mut_df.shape[0]:,} patients × {mut_df.shape[1]:,} features")

    return expr_df, meth_df, mut_df, meta_df


# ── Step 2: Integrate Modalities ─────────────────────────────────────────────

def integrate_modalities(expr_df, meth_df, mut_df, meta_df):
    """Join three modalities by patient ID (inner join)."""
    print("\n" + "=" * 70)
    print("STEP 2: Integrating Modalities (Three-Way Intersection)")
    print("=" * 70)

    patient_cancer = dict(zip(meta_df["patient_id"], meta_df["cancer_type"]))

    common_patients = sorted(
        set(expr_df.index) & set(mut_df.index) & set(meth_df.index)
    )
    print(f"  Patients with all 3 modalities: {len(common_patients)}")

    # Prefix mutation and methylation columns to avoid collisions
    mut_sub = mut_df.loc[common_patients].copy()
    mut_sub.columns = [f"mut_{c}" for c in mut_sub.columns]

    meth_sub = meth_df.loc[common_patients].copy()
    meth_sub.columns = [f"meth_{c}" for c in meth_sub.columns]

    # Concatenate: expression (2000) + mutations (63) + methylation (2000) = 4063
    feature_df = pd.concat(
        [expr_df.loc[common_patients], mut_sub, meth_sub], axis=1
    )

    labels = pd.Series(
        [patient_cancer[pid] for pid in feature_df.index],
        index=feature_df.index,
        name="cancer_type",
    )

    print(f"  Integrated feature matrix: {feature_df.shape[0]:,} × {feature_df.shape[1]:,}")
    print(f"  Cancer types: {sorted(labels.unique())}")
    print(f"\n  Per-type counts (before balancing):")
    for ct, count in labels.value_counts().sort_index().items():
        print(f"    {ct}: {count}")

    return feature_df, labels


# ── Step 3: Balance Classes ──────────────────────────────────────────────────

def balance_dataset(feature_df, labels):
    """Down-sample each class to the size of the smallest class."""
    print("\n" + "=" * 70)
    print("STEP 3: Balancing Classes (Down-Sampling)")
    print("=" * 70)

    class_counts = labels.value_counts()
    min_count = class_counts.min()
    min_class = class_counts.idxmin()

    print(f"  Smallest class: {min_class} with {min_count} samples")
    print(f"  Down-sampling all classes to {min_count} samples each")

    rng = np.random.RandomState(RANDOM_STATE)
    sampled_indices = []
    for ct in sorted(class_counts.index):
        ct_indices = labels[labels == ct].index.tolist()
        chosen = rng.choice(ct_indices, size=min_count, replace=False)
        sampled_indices.extend(chosen)

    feature_bal = feature_df.loc[sampled_indices]
    labels_bal = labels.loc[sampled_indices]

    print(f"  Balanced dataset: {len(labels_bal)} samples ({min_count} × {labels_bal.nunique()} types)")

    return feature_bal, labels_bal


# ── Step 4: Train/Test Split ─────────────────────────────────────────────────

def split_data(feature_df, labels):
    """Stratified 80/20 train/test split."""
    print("\n" + "=" * 70)
    print("STEP 4: Train/Test Split (80/20, Stratified)")
    print("=" * 70)

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        feature_df, labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    print(f"  Training set: {X_train_df.shape[0]} samples")
    print(f"  Test set:     {X_test_df.shape[0]} samples")

    # Scale features using training set statistics only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df.values)
    X_test = scaler.transform(X_test_df.values)

    # Encode labels
    le = LabelEncoder()
    le.fit(sorted(y_train.unique()))
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    print(f"  Feature scaling: StandardScaler (fit on training set only)")
    print(f"  Classes: {list(le.classes_)}")

    return X_train, X_test, y_train_enc, y_test_enc, le, scaler, feature_df.columns.tolist()


# ── Step 5: Train Models ─────────────────────────────────────────────────────

def optimize_lightgbm(X_train, y_train_enc, n_classes):
    """Bayesian hyperparameter optimization for LightGBM via Optuna."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "objective": "multiclass",
            "num_class": n_classes,
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "num_leaves": trial.suggest_int("num_leaves", 20, 80),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
            "verbosity": -1,
            "random_state": RANDOM_STATE,
        }
        model = lgb.LGBMClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train_enc):
            model.fit(X_train[train_idx], y_train_enc[train_idx])
            preds = model.predict(X_train[val_idx])
            scores.append(balanced_accuracy_score(y_train_enc[val_idx], preds))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    print(f"  Best Optuna trial: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    best_params = study.best_params
    best_params.update({
        "objective": "multiclass",
        "num_class": n_classes,
        "verbosity": -1,
        "random_state": RANDOM_STATE,
    })
    return lgb.LGBMClassifier(**best_params)


def get_models(X_train, y_train_enc, n_classes, skip_optuna=False):
    """Define all models. Optionally optimize LightGBM with Optuna."""
    print("\n" + "=" * 70)
    print("STEP 5: Training Models")
    print("=" * 70)

    if skip_optuna:
        print("  [Skipping Optuna — using fixed LightGBM hyperparameters]")
        lgb_model = lgb.LGBMClassifier(
            objective="multiclass", num_class=n_classes,
            n_estimators=500, num_leaves=45, max_depth=7,
            learning_rate=0.05, min_child_samples=25,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=2.5, reg_lambda=3.2,
            verbosity=-1, random_state=RANDOM_STATE,
        )
    else:
        print("  Running Bayesian hyperparameter optimization (Optuna, 20 trials)...")
        lgb_model = optimize_lightgbm(X_train, y_train_enc, n_classes)

    models = {
        "LightGBM": lgb_model,
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=0,
            eval_metric="mlogloss",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500, max_depth=None,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "LogisticRegression": LogisticRegression(
            solver="lbfgs", max_iter=2000, C=1.0,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }

    return models


# ── Step 6: Evaluate ─────────────────────────────────────────────────────────

def evaluate_models(models, X_train, X_test, y_train_enc, y_test_enc, le):
    """Train each model and evaluate on the held-out test set."""
    print("\n" + "=" * 70)
    print("STEP 6: Evaluating on Held-Out Test Set")
    print("=" * 70)

    results = {}
    best_model = None
    best_acc = 0

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train_enc)
        train_time = time.time() - t0

        y_pred = model.predict(X_test)
        bal_acc = balanced_accuracy_score(y_test_enc, y_pred)
        report = classification_report(
            y_test_enc, y_pred, target_names=le.classes_, output_dict=True
        )
        cm = confusion_matrix(y_test_enc, y_pred)

        results[name] = {
            "test_balanced_accuracy": float(bal_acc),
            "train_time_seconds": round(train_time, 1),
            "classification_report": {
                ct: {k: round(v, 4) for k, v in report[ct].items()}
                for ct in le.classes_
            },
            "confusion_matrix": cm.tolist(),
        }

        if bal_acc > best_acc:
            best_acc = bal_acc
            best_model = (name, model)

        print(f"\n  {name}:")
        print(f"    Balanced accuracy: {bal_acc:.4f}  (train time: {train_time:.1f}s)")
        # Per-class summary
        for ct in le.classes_:
            p = report[ct]["precision"]
            r = report[ct]["recall"]
            f1 = report[ct]["f1-score"]
            print(f"    {ct}: P={p:.3f} R={r:.3f} F1={f1:.3f}")

    return results, best_model


# ── Step 7: Cross-Validation ─────────────────────────────────────────────────

def cross_validate_models(models, X_train, y_train_enc):
    """Stratified 5-fold cross-validation on the training set."""
    print("\n" + "=" * 70)
    print("STEP 7: Stratified 5-Fold Cross-Validation")
    print("=" * 70)

    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}

    for name, model in models.items():
        scores = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train_enc)):
            # Clone model for each fold
            from sklearn.base import clone
            fold_model = clone(model)
            fold_model.fit(X_train[train_idx], y_train_enc[train_idx])
            preds = fold_model.predict(X_train[val_idx])
            score = balanced_accuracy_score(y_train_enc[val_idx], preds)
            scores.append(score)

        cv_results[name] = {
            "cv_scores": [round(s, 4) for s in scores],
            "cv_mean": round(float(np.mean(scores)), 4),
            "cv_std": round(float(np.std(scores)), 4),
        }
        print(f"  {name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}  {[f'{s:.3f}' for s in scores]}")

    return cv_results


# ── Step 8: SHAP Analysis ────────────────────────────────────────────────────

def run_shap_analysis(model, X_test, le, feature_names, output_dir):
    """SHAP feature importance for the best model."""
    print("\n" + "=" * 70)
    print("STEP 8: SHAP Feature Importance Analysis")
    print("=" * 70)

    try:
        import shap
    except ImportError:
        print("  SHAP not available — skipping. Install with: pip install shap")
        return None

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Mean absolute SHAP across all classes and samples
        if isinstance(shap_values, list):
            mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            mean_abs = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False)

        shap_path = output_dir / "shap_feature_importance.csv"
        importance_df.to_csv(shap_path, index=False)
        print(f"  Saved SHAP importance to {shap_path}")

        print(f"\n  Top 20 features:")
        for i, row in importance_df.head(20).iterrows():
            print(f"    {row['feature']:30s}  SHAP={row['mean_abs_shap']:.6f}")

        return importance_df

    except Exception as e:
        print(f"  SHAP analysis failed: {e}")
        return None


# ── Step 9: Save Results ─────────────────────────────────────────────────────

def save_results(results, cv_results, le, scaler, best_model, feature_names, output_dir):
    """Save all results, the best model, and metadata."""
    print("\n" + "=" * 70)
    print("STEP 9: Saving Results")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # Merge test and CV results
    combined = {}
    for name in results:
        combined[name] = {**results[name], **cv_results.get(name, {})}

    with open(output_dir / "model_results.json", "w") as f:
        json.dump(combined, f, indent=2)
    print(f"  Saved model_results.json")

    # Dataset info
    best_name, _ = best_model
    info = {
        "pipeline": "run_full_pipeline.py",
        "n_features": len(feature_names),
        "n_classes": len(le.classes_),
        "classes": list(le.classes_),
        "best_model": best_name,
        "best_test_accuracy": combined[best_name]["test_balanced_accuracy"],
        "best_cv_mean": combined[best_name].get("cv_mean"),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": N_CV_FOLDS,
    }
    with open(output_dir / "pipeline_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Saved pipeline_info.json")

    # Save best model
    import joblib
    joblib.dump(best_model[1], output_dir / "best_model.pkl")
    joblib.dump(scaler, output_dir / "scaler.pkl")
    joblib.dump(le, output_dir / "label_encoder.pkl")
    print(f"  Saved best_model.pkl, scaler.pkl, label_encoder.pkl")

    # Feature names
    with open(output_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f)
    print(f"  Saved feature_names.json ({len(feature_names)} features)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Oncura full reproducible pipeline")
    parser.add_argument("--skip-optuna", action="store_true",
                        help="Skip Bayesian optimization (use fixed hyperparameters)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory for results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    t_start = time.time()

    print("\n" + "#" * 70)
    print("#  ONCURA — Full Reproducible Pipeline")
    print("#  Multi-Modal Cancer Classification on Real TCGA Data")
    print("#" * 70)

    # Step 1: Load
    expr_df, meth_df, mut_df, meta_df = load_features()

    # Step 2: Integrate
    feature_df, labels = integrate_modalities(expr_df, meth_df, mut_df, meta_df)

    # Step 3: Balance
    feature_bal, labels_bal = balance_dataset(feature_df, labels)

    # Step 4: Split
    X_train, X_test, y_train_enc, y_test_enc, le, scaler, feature_names = \
        split_data(feature_bal, labels_bal)

    n_classes = len(le.classes_)

    # Step 5: Train
    models = get_models(X_train, y_train_enc, n_classes, skip_optuna=args.skip_optuna)

    # Step 6: Evaluate
    results, best_model = evaluate_models(
        models, X_train, X_test, y_train_enc, y_test_enc, le
    )

    # Step 7: Cross-validate
    cv_results = cross_validate_models(models, X_train, y_train_enc)

    # Step 8: SHAP (always run on LightGBM since TreeExplainer supports it)
    lgb_model = models["LightGBM"]
    shap_df = run_shap_analysis(lgb_model, X_test, le, feature_names, output_dir)

    # Step 9: Save
    save_results(results, cv_results, le, scaler, best_model, feature_names, output_dir)

    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE — {elapsed:.0f}s total")
    print(f"Best model: {best_name} — {results[best_name]['test_balanced_accuracy']:.4f} balanced accuracy")
    print(f"Results saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
