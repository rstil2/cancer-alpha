#!/usr/bin/env python3
"""Step 4: Integrate features, build balanced dataset, train models, evaluate.

Joins expression + mutation + methylation features by patient ID,
creates balanced dataset, trains multiple classifiers with hyperparameter
tuning, evaluates on held-out test set, and runs SHAP analysis.
"""

import json
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import lightgbm as lgb
import shap
import xgboost as xgb

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "real_model_results"

RANDOM_STATE = 42
N_CV_FOLDS = 5
OPTUNA_TRIALS = 20


def load_features():
    """Load and integrate expression + mutation + methylation features."""
    print("Loading features...")

    expr_path = OUTPUT_DIR / "expression_features.pkl"
    mut_path = OUTPUT_DIR / "mutation_features.pkl"
    meth_path = OUTPUT_DIR / "methylation_features.pkl"
    expr_meta_path = OUTPUT_DIR / "expression_metadata.csv"

    if not expr_path.exists():
        print("ERROR: expression_features.pkl not found. Run step2 first.")
        sys.exit(1)

    expr_df = pd.read_pickle(expr_path)
    expr_meta = pd.read_csv(expr_meta_path)
    print(f"  Expression: {expr_df.shape}")

    # Build patient -> cancer_type mapping from expression metadata
    patient_cancer = dict(zip(expr_meta["patient_id"], expr_meta["cancer_type"]))

    # Start with expression features
    feature_df = expr_df.copy()
    common_patients = set(feature_df.index)

    # Add mutation features if available
    if mut_path.exists():
        mut_df = pd.read_pickle(mut_path)
        print(f"  Mutations: {mut_df.shape}")
        common_patients &= set(mut_df.index)
    else:
        print("  Mutations: not found")
        mut_df = None

    # Add methylation features if available
    if meth_path.exists():
        meth_df = pd.read_pickle(meth_path)
        print(f"  Methylation: {meth_df.shape}")
        common_patients &= set(meth_df.index)
    else:
        print("  Methylation: not found")
        meth_df = None

    # Intersect all modalities
    common_patients = sorted(common_patients)
    print(f"  Patients with all available modalities: {len(common_patients)}")

    # Build integrated feature matrix using pd.concat (avoids fragmentation)
    frames = [feature_df.loc[common_patients]]

    if mut_df is not None:
        mut_sub = mut_df.loc[common_patients].copy()
        mut_sub.columns = [f"mut_{c}" for c in mut_sub.columns]
        frames.append(mut_sub)

    if meth_df is not None:
        meth_sub = meth_df.loc[common_patients].copy()
        meth_sub.columns = [f"meth_{c}" for c in meth_sub.columns]
        frames.append(meth_sub)

    feature_df = pd.concat(frames, axis=1)

    # Add cancer type labels
    labels = [patient_cancer[pid] for pid in feature_df.index]

    print(f"\n  Integrated feature matrix: {feature_df.shape}")
    return feature_df, labels


def balance_dataset(feature_df, labels, random_state=RANDOM_STATE):
    """Balance dataset by sampling N per class where N = min class size."""
    df = feature_df.copy()
    df["_label"] = labels

    class_counts = df["_label"].value_counts()
    min_count = class_counts.min()
    print(f"\n  Class sizes before balancing: {dict(class_counts)}")
    print(f"  Balancing to {min_count} per class")

    # Sample min_count per class
    sampled_indices = []
    for label in class_counts.index:
        group = df[df["_label"] == label]
        sampled = group.sample(n=min_count, random_state=random_state)
        sampled_indices.append(sampled.index)

    balanced = df.loc[np.concatenate(sampled_indices)].copy()
    labels_balanced = balanced.pop("_label").tolist()
    return balanced, labels_balanced


def train_test_split_balanced(X, y, test_size=0.2, random_state=RANDOM_STATE):
    """Stratified train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    return X_train, X_test, y_train, y_test


def optimize_lightgbm(X_train, y_train, le, n_trials=OPTUNA_TRIALS):
    """Bayesian hyperparameter optimization for LightGBM."""
    print(f"\n  Optimizing LightGBM ({n_trials} trials)...")
    y_encoded = le.transform(y_train)
    n_classes = len(le.classes_)

    def objective(trial):
        params = {
            "objective": "multiclass",
            "num_class": n_classes,
            "metric": "multi_logloss",
            "verbosity": -1,
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": RANDOM_STATE,
        }

        model = lgb.LGBMClassifier(**params)
        cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(
            model, X_train, y_encoded, cv=cv, scoring="balanced_accuracy",
        )
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"  Best CV balanced accuracy: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params


def train_and_evaluate_models(X_train, X_test, y_train, y_test, le):
    """Train multiple models and evaluate on test set."""
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    # Optimize LightGBM
    best_lgb_params = optimize_lightgbm(X_train, y_train, le)

    models = {
        "LightGBM": lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(le.classes_),
            verbosity=-1,
            random_state=RANDOM_STATE,
            **best_lgb_params,
        ),
        "XGBoost": xgb.XGBClassifier(
            objective="multi:softprob",
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ),
    }

    results = {}
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        print(f"\n  Training {name}...")

        # Cross-validation on training set
        cv_scores = cross_val_score(
            model, X_train, y_train_enc, cv=cv, scoring="balanced_accuracy",
        )

        # Train on full training set, evaluate on test
        model.fit(X_train, y_train_enc)
        y_pred = model.predict(X_test)
        test_bal_acc = balanced_accuracy_score(y_test_enc, y_pred)

        report = classification_report(
            y_test_enc, y_pred, target_names=le.classes_, output_dict=True,
        )
        cm = confusion_matrix(y_test_enc, y_pred)

        results[name] = {
            "model": model,
            "cv_scores": cv_scores.tolist(),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "test_balanced_accuracy": float(test_bal_acc),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }

        print(f"    CV balanced accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"    Test balanced accuracy: {test_bal_acc:.4f}")

    return results


def run_shap_analysis(model, X_test, feature_names, le, top_n=30):
    """Run SHAP analysis on the best model."""
    print("\n  Running SHAP analysis...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Global feature importance (mean absolute SHAP across all classes)
    if isinstance(shap_values, list):
        # Multi-class: list of arrays
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    elif shap_values.ndim == 3:
        # Multi-class: (n_samples, n_features, n_classes)
        mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    print(f"\n  Top {top_n} most important features:")
    for _, row in importance_df.head(top_n).iterrows():
        print(f"    {row['feature']}: {row['mean_abs_shap']:.4f}")

    return importance_df, shap_values


def main():
    print("=" * 60)
    print("Step 4: Dataset integration, model training, evaluation")
    print("=" * 60)

    # Load and integrate features
    feature_df, labels = load_features()

    # Balance dataset
    feature_df, labels = balance_dataset(feature_df, labels)

    # Encode labels
    le = LabelEncoder()
    le.fit(sorted(set(labels)))
    print(f"\n  Cancer types ({len(le.classes_)}): {list(le.classes_)}")

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_df.values)
    feature_names = list(feature_df.columns)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split_balanced(
        X, labels, test_size=0.2,
    )
    print(f"\n  Train: {X_train.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")

    # Train and evaluate models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test, le)

    # Find best model
    best_name = max(results, key=lambda k: results[k]["test_balanced_accuracy"])
    best_result = results[best_name]
    best_model = best_result["model"]
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_name}")
    print(f"  CV balanced accuracy: {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}")
    print(f"  Test balanced accuracy: {best_result['test_balanced_accuracy']:.4f}")
    print(f"{'='*60}")

    # Per-class results for best model
    print(f"\nPer-class results ({best_name}):")
    report = best_result["classification_report"]
    for ct in le.classes_:
        r = report[ct]
        print(f"  {ct}: precision={r['precision']:.3f} recall={r['recall']:.3f} f1={r['f1-score']:.3f}")

    # SHAP analysis on best model
    importance_df, shap_values = run_shap_analysis(best_model, X_test, feature_names, le)

    # Save all results
    print("\nSaving results...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save results summary (without model objects)
    results_summary = {}
    for name, r in results.items():
        results_summary[name] = {k: v for k, v in r.items() if k != "model"}
    with open(OUTPUT_DIR / "model_results.json", "w") as f:
        json.dump(results_summary, f, indent=2, default=str)

    # Save best model
    with open(OUTPUT_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    # Save scaler and label encoder
    with open(OUTPUT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(OUTPUT_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    # Save feature importance
    importance_df.to_csv(OUTPUT_DIR / "feature_importance_shap.csv", index=False)

    # Save feature names
    with open(OUTPUT_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    # Save dataset info
    dataset_info = {
        "total_samples": X_train.shape[0] + X_test.shape[0],
        "train_samples": X_train.shape[0],
        "test_samples": X_test.shape[0],
        "n_features": X_train.shape[1],
        "n_classes": len(le.classes_),
        "classes": list(le.classes_),
        "samples_per_class": X_train.shape[0] // len(le.classes_),
        "best_model": best_name,
        "best_cv_accuracy": best_result["cv_mean"],
        "best_cv_std": best_result["cv_std"],
        "best_test_accuracy": best_result["test_balanced_accuracy"],
    }
    with open(OUTPUT_DIR / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"\nAll results saved to {OUTPUT_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
