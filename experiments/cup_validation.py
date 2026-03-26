#!/usr/bin/env python3
"""CUP (Cancer of Unknown Primary) Validation Experiment.

Simulates the real clinical CUP scenario:
  - Train classifier on known cancer samples (8 types)
  - Classify held-out "unknown" samples and predict their primary type
  - Evaluate accuracy, confidence calibration, and per-type performance
  - Repeat with 10 random train/test splits for robustness

This demonstrates the classifier's utility for CUP diagnosis:
given a tumor sample of unknown origin, how accurately can we
identify the cancer type using multi-modal molecular features?
"""

import json
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "real_model_results"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "cup_results"

N_REPEATS = 10
TEST_SIZE = 0.2


def load_integrated_features():
    """Load and integrate expression + mutation + methylation features."""
    print("Loading features...", flush=True)

    expr_df = pd.read_pickle(DATA_DIR / "expression_features.pkl")
    expr_meta = pd.read_csv(DATA_DIR / "expression_metadata.csv")
    mut_df = pd.read_pickle(DATA_DIR / "mutation_features.pkl")
    meth_df = pd.read_pickle(DATA_DIR / "methylation_features.pkl")

    patient_cancer = dict(zip(expr_meta["patient_id"], expr_meta["cancer_type"]))

    common_patients = sorted(
        set(expr_df.index) & set(mut_df.index) & set(meth_df.index)
    )

    mut_sub = mut_df.loc[common_patients].copy()
    mut_sub.columns = [f"mut_{c}" for c in mut_sub.columns]
    meth_sub = meth_df.loc[common_patients].copy()
    meth_sub.columns = [f"meth_{c}" for c in meth_sub.columns]

    feature_df = pd.concat(
        [expr_df.loc[common_patients], mut_sub, meth_sub], axis=1
    )

    labels = pd.Series(
        [patient_cancer[pid] for pid in feature_df.index],
        index=feature_df.index,
        name="cancer_type",
    )

    print(f"  {feature_df.shape[0]} samples, {feature_df.shape[1]} features, "
          f"{labels.nunique()} cancer types", flush=True)
    return feature_df, labels


def balance_dataset(feature_df, labels, random_state):
    """Balance dataset by down-sampling to smallest class."""
    class_counts = labels.value_counts()
    min_count = class_counts.min()
    sampled = []
    rng = np.random.RandomState(random_state)
    for ct in class_counts.index:
        idx = labels[labels == ct].index
        chosen = rng.choice(idx, size=min_count, replace=False)
        sampled.extend(chosen)
    return feature_df.loc[sampled], labels.loc[sampled]


def run_cup_validation():
    """Run multiple CUP hold-out validations."""
    feature_df, labels = load_integrated_features()

    # Balance dataset first
    feature_df, labels = balance_dataset(feature_df, labels, random_state=42)
    n_per_class = labels.value_counts().min()
    cancer_types = sorted(labels.unique())
    n_types = len(cancer_types)

    print(f"\n  Balanced: {len(labels)} samples, {n_per_class} per type", flush=True)
    print(f"  Cancer types: {cancer_types}", flush=True)
    print(f"  Running {N_REPEATS} repeated CUP validations...\n", flush=True)

    le = LabelEncoder()
    le.fit(cancer_types)

    all_repeat_results = []
    all_predictions = []  # For calibration analysis

    for rep in range(N_REPEATS):
        seed = 42 + rep
        print(f"  Repeat {rep+1}/{N_REPEATS} (seed={seed})...", flush=True)

        # Stratified train/test split
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            feature_df, labels, test_size=TEST_SIZE,
            random_state=seed, stratify=labels,
        )

        y_train_enc = le.transform(y_train)
        y_test_enc = le.transform(y_test)

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_df.values)
        X_test = scaler.transform(X_test_df.values)

        # Train LightGBM (fixed hyperparameters — no Optuna for speed)
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=n_types,
            n_estimators=500,
            num_leaves=45,
            max_depth=7,
            learning_rate=0.05,
            min_child_samples=25,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=2.5,
            reg_lambda=3.2,
            verbosity=-1,
            random_state=seed,
        )
        model.fit(X_train, y_train_enc)

        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Overall metrics
        bal_acc = balanced_accuracy_score(y_test_enc, y_pred)

        # Top-3 accuracy
        top3_idx = np.argsort(y_proba, axis=1)[:, -3:]
        top3_correct = sum(y_test_enc[i] in top3_idx[i] for i in range(len(y_test_enc)))
        top3_acc = top3_correct / len(y_test_enc)

        # Per-class metrics
        report = classification_report(
            y_test_enc, y_pred, target_names=le.classes_, output_dict=True,
        )

        # Confidence stats
        max_proba = y_proba.max(axis=1)
        correct_mask = y_pred == y_test_enc

        print(f"    Balanced accuracy: {bal_acc:.4f}, Top-3: {top3_acc:.4f}, "
              f"Mean confidence: {max_proba.mean():.4f}", flush=True)

        # Collect per-sample predictions for calibration
        for i in range(len(y_test_enc)):
            all_predictions.append({
                "repeat": rep,
                "true_label": le.classes_[y_test_enc[i]],
                "pred_label": le.classes_[y_pred[i]],
                "confidence": float(max_proba[i]),
                "correct": bool(correct_mask[i]),
                "true_proba": float(y_proba[i, y_test_enc[i]]),
            })

        all_repeat_results.append({
            "repeat": rep,
            "seed": seed,
            "balanced_accuracy": float(bal_acc),
            "top3_accuracy": float(top3_acc),
            "mean_confidence": float(max_proba.mean()),
            "mean_confidence_correct": float(max_proba[correct_mask].mean()) if correct_mask.any() else 0,
            "mean_confidence_incorrect": float(max_proba[~correct_mask].mean()) if (~correct_mask).any() else 0,
            "confusion_matrix": confusion_matrix(y_test_enc, y_pred).tolist(),
            "per_class": {ct: report[ct] for ct in le.classes_},
        })

    return all_repeat_results, all_predictions, cancer_types, le


def compute_summary(all_repeat_results, all_predictions, cancer_types):
    """Compute aggregate CUP validation metrics."""
    bal_accs = [r["balanced_accuracy"] for r in all_repeat_results]
    top3_accs = [r["top3_accuracy"] for r in all_repeat_results]

    pred_df = pd.DataFrame(all_predictions)

    # Confidence calibration bins
    bins = np.arange(0, 1.05, 0.1)
    calibration = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (pred_df["confidence"] >= lo) & (pred_df["confidence"] < hi)
        subset = pred_df[mask]
        if len(subset) > 0:
            calibration.append({
                "bin_center": (lo + hi) / 2,
                "mean_confidence": float(subset["confidence"].mean()),
                "accuracy": float(subset["correct"].mean()),
                "count": int(len(subset)),
            })

    # High-confidence accuracy
    high_conf_mask = pred_df["confidence"] >= 0.9
    high_conf_acc = pred_df[high_conf_mask]["correct"].mean() if high_conf_mask.any() else 0
    high_conf_pct = high_conf_mask.mean()

    # Per-cancer aggregate across repeats
    per_cancer = {}
    for ct in cancer_types:
        ct_preds = pred_df[pred_df["true_label"] == ct]
        per_cancer[ct] = {
            "n_predictions": len(ct_preds),
            "accuracy": float(ct_preds["correct"].mean()),
            "mean_confidence": float(ct_preds["confidence"].mean()),
        }

    summary = {
        "n_repeats": len(all_repeat_results),
        "balanced_accuracy_mean": float(np.mean(bal_accs)),
        "balanced_accuracy_std": float(np.std(bal_accs)),
        "balanced_accuracy_min": float(np.min(bal_accs)),
        "balanced_accuracy_max": float(np.max(bal_accs)),
        "top3_accuracy_mean": float(np.mean(top3_accs)),
        "top3_accuracy_std": float(np.std(top3_accs)),
        "high_confidence_threshold": 0.9,
        "high_confidence_accuracy": float(high_conf_acc),
        "high_confidence_fraction": float(high_conf_pct),
        "calibration": calibration,
        "per_cancer": per_cancer,
        "total_predictions": len(pred_df),
        "overall_accuracy": float(pred_df["correct"].mean()),
    }

    print(f"\n{'='*70}", flush=True)
    print("CUP VALIDATION SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  {len(all_repeat_results)} repeated hold-out validations", flush=True)
    print(f"  Balanced accuracy: {np.mean(bal_accs):.4f} ± {np.std(bal_accs):.4f} "
          f"(range: {np.min(bal_accs):.4f}–{np.max(bal_accs):.4f})", flush=True)
    print(f"  Top-3 accuracy:    {np.mean(top3_accs):.4f} ± {np.std(top3_accs):.4f}", flush=True)
    print(f"  High-confidence (≥0.9) predictions: {high_conf_pct*100:.1f}% of samples", flush=True)
    print(f"  High-confidence accuracy: {high_conf_acc*100:.1f}%", flush=True)
    print(f"\n  Per-cancer accuracy:", flush=True)
    for ct in cancer_types:
        pc = per_cancer[ct]
        print(f"    {ct}: {pc['accuracy']*100:.1f}% (confidence: {pc['mean_confidence']:.3f})", flush=True)

    return summary


def plot_cup_results(all_repeat_results, all_predictions, summary, cancer_types, le, output_dir):
    """Generate CUP validation figures."""
    os.makedirs(output_dir, exist_ok=True)
    pred_df = pd.DataFrame(all_predictions)

    # --- Figure 1: Accuracy across repeats ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    bal_accs = [r["balanced_accuracy"] * 100 for r in all_repeat_results]
    top3_accs = [r["top3_accuracy"] * 100 for r in all_repeat_results]

    ax = axes[0]
    ax.bar(range(len(bal_accs)), bal_accs, color="#2196F3", alpha=0.8, label="Top-1")
    ax.bar(range(len(top3_accs)), top3_accs, color="#4CAF50", alpha=0.4, label="Top-3")
    ax.set_xlabel("Repeat", fontsize=11)
    ax.set_ylabel("Balanced Accuracy (%)", fontsize=11)
    ax.set_title("CUP Classification Across 10 Hold-Out Splits", fontsize=12, fontweight="bold")
    ax.set_ylim(80, 102)
    ax.legend()
    ax.axhline(y=np.mean(bal_accs), color="blue", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    # Aggregate confusion matrix
    ax = axes[1]
    total_cm = np.zeros((len(cancer_types), len(cancer_types)))
    for r in all_repeat_results:
        total_cm += np.array(r["confusion_matrix"])
    # Normalize per row
    row_sums = total_cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_pct = total_cm / row_sums * 100

    short = [ct.replace("TCGA-", "") for ct in cancer_types]
    sns.heatmap(cm_pct, annot=True, fmt=".1f", xticklabels=short, yticklabels=short,
                cmap="Blues", vmin=0, vmax=100, ax=ax,
                cbar_kws={"label": "% predictions"})
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title("Aggregate Confusion Matrix\n(10 repeats, % normalized)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_dir / "cup_validation_accuracy.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "cup_validation_accuracy.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: cup_validation_accuracy.png/pdf", flush=True)

    # --- Figure 2: Confidence calibration ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Calibration plot
    ax = axes[0]
    cal = summary["calibration"]
    confs = [c["mean_confidence"] for c in cal]
    accs = [c["accuracy"] for c in cal]
    counts = [c["count"] for c in cal]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.bar(confs, accs, width=0.08, color="#FF9800", alpha=0.8, edgecolor="white")
    ax.set_xlabel("Mean Predicted Confidence", fontsize=11)
    ax.set_ylabel("Observed Accuracy", fontsize=11)
    ax.set_title("Confidence Calibration", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    # Confidence histogram (correct vs incorrect)
    ax = axes[1]
    correct = pred_df[pred_df["correct"]]["confidence"]
    incorrect = pred_df[~pred_df["correct"]]["confidence"]
    ax.hist(correct, bins=20, alpha=0.7, color="#4CAF50", label=f"Correct (n={len(correct)})", density=True)
    ax.hist(incorrect, bins=20, alpha=0.7, color="#F44336", label=f"Incorrect (n={len(incorrect)})", density=True)
    ax.set_xlabel("Prediction Confidence", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Confidence Distribution", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # Per-cancer accuracy
    ax = axes[2]
    per_c = summary["per_cancer"]
    types_sorted = sorted(per_c.keys(), key=lambda x: per_c[x]["accuracy"], reverse=True)
    accs = [per_c[ct]["accuracy"] * 100 for ct in types_sorted]
    short = [ct.replace("TCGA-", "") for ct in types_sorted]
    colors = ["#4CAF50" if a >= 95 else "#FF9800" if a >= 90 else "#F44336" for a in accs]
    ax.barh(range(len(accs)), accs, color=colors, edgecolor="white")
    ax.set_yticks(range(len(accs)))
    ax.set_yticklabels(short, fontsize=10)
    ax.set_xlabel("Accuracy (%)", fontsize=11)
    ax.set_title("Per-Cancer CUP Accuracy\n(aggregated over 10 repeats)", fontsize=12, fontweight="bold")
    ax.set_xlim(80, 102)
    ax.grid(axis="x", alpha=0.3)
    for i, a in enumerate(accs):
        ax.text(a + 0.3, i, f"{a:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "cup_calibration_analysis.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "cup_calibration_analysis.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: cup_calibration_analysis.png/pdf", flush=True)


def main():
    print("=" * 70, flush=True)
    print("CUP VALIDATION: Repeated Hold-Out Cancer Type Classification", flush=True)
    print("=" * 70, flush=True)

    all_repeat_results, all_predictions, cancer_types, le = run_cup_validation()
    summary = compute_summary(all_repeat_results, all_predictions, cancer_types)

    print("\nGenerating figures...", flush=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_cup_results(all_repeat_results, all_predictions, summary, cancer_types, le, OUTPUT_DIR)

    # Save results
    with open(OUTPUT_DIR / "cup_validation_results.json", "w") as f:
        json.dump(all_repeat_results, f, indent=2)

    with open(OUTPUT_DIR / "cup_validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(OUTPUT_DIR / "cup_predictions.json", "w") as f:
        json.dump(all_predictions, f, indent=2)

    print(f"\nAll results saved to {OUTPUT_DIR}/", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
