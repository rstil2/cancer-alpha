#!/usr/bin/env python3
"""HISTORICAL — SciReports-era CUP leave-one-out simulation. Not a JBI public claim.

See docs/MANUSCRIPT_ARCHIVE.md. Do not cite 97.6% CUP in README or demos.

Original: Cancer of Unknown Primary (CUP) Simulation via Leave-One-Cancer-Out.
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
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
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

RANDOM_STATE = 42
N_CV_FOLDS = 5


def load_integrated_features():
    """Load and integrate expression + mutation + methylation features.

    Replicates the data loading from src/pipeline/step4_train_evaluate.py.
    """
    print("Loading features...")

    expr_df = pd.read_pickle(DATA_DIR / "expression_features.pkl")
    expr_meta = pd.read_csv(DATA_DIR / "expression_metadata.csv")
    mut_df = pd.read_pickle(DATA_DIR / "mutation_features.pkl")
    meth_df = pd.read_pickle(DATA_DIR / "methylation_features.pkl")

    patient_cancer = dict(zip(expr_meta["patient_id"], expr_meta["cancer_type"]))

    # Intersect patients with all three modalities
    common_patients = sorted(
        set(expr_df.index) & set(mut_df.index) & set(meth_df.index)
    )
    print(f"  Patients with all 3 modalities: {len(common_patients)}")

    # Concatenate features
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

    print(f"  Integrated feature matrix: {feature_df.shape}")
    print(f"  Cancer types: {sorted(labels.unique())}")
    print(f"  Samples per type:")
    for ct, count in labels.value_counts().sort_index().items():
        print(f"    {ct}: {count}")

    return feature_df, labels


def balance_dataset(feature_df, labels, random_state=RANDOM_STATE):
    """Balance dataset by sampling N per class where N = min class size."""
    class_counts = labels.value_counts()
    min_count = class_counts.min()

    sampled_indices = []
    for ct in class_counts.index:
        group_idx = labels[labels == ct].index
        rng = np.random.RandomState(random_state)
        chosen = rng.choice(group_idx, size=min_count, replace=False)
        sampled_indices.extend(chosen)

    return feature_df.loc[sampled_indices], labels.loc[sampled_indices]


def run_cup_simulation(feature_df, labels):
    """Run leave-one-cancer-out CUP simulation.

    For each of the 8 cancer types:
    1. Hold out ALL samples of that type
    2. Train LightGBM on the remaining 7 types (balanced)
    3. Predict the held-out samples
    4. Record top-1 accuracy, top-3 accuracy, confidence, misclassification
    """
    cancer_types = sorted(labels.unique())
    n_types = len(cancer_types)

    print(f"\n{'='*70}")
    print(f"LEAVE-ONE-CANCER-OUT CUP SIMULATION ({n_types} cancer types)")
    print(f"{'='*70}")

    all_results = {}

    for i, held_out in enumerate(cancer_types):
        print(f"\n--- Iteration {i+1}/{n_types}: Holding out {held_out} ---")

        # Split: training = all other types, test = held-out type
        train_mask = labels != held_out
        test_mask = labels == held_out

        X_train_raw = feature_df[train_mask]
        y_train_raw = labels[train_mask]
        X_test_raw = feature_df[test_mask]
        y_test_raw = labels[test_mask]

        # Balance training set (remaining 7 types)
        X_train_bal, y_train_bal = balance_dataset(X_train_raw, y_train_raw)

        print(f"  Training: {len(y_train_bal)} samples ({len(y_train_bal.unique())} types)")
        print(f"  Test (held-out {held_out}): {len(y_test_raw)} samples")

        # Encode labels — include all 8 types so predictions can map correctly
        le = LabelEncoder()
        le.fit(cancer_types)
        y_train_enc = le.transform(y_train_bal)
        y_test_enc = le.transform(y_test_raw)
        held_out_idx = le.transform([held_out])[0]

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_bal.values)
        X_test = scaler.transform(X_test_raw.values)

        # Train LightGBM with sensible defaults (matching main pipeline)
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
            random_state=RANDOM_STATE,
        )

        # Cross-validation on training set (7 types)
        cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(
            model, X_train, y_train_enc, cv=cv, scoring="balanced_accuracy"
        )
        print(f"  Training CV balanced accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Train on full training set
        model.fit(X_train, y_train_enc)

        # Predict held-out cancer type
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Top-1 accuracy: does the model correctly identify the held-out cancer?
        top1_correct = (y_pred == held_out_idx).sum()
        top1_accuracy = top1_correct / len(y_test_enc)

        # Top-3 accuracy: is the true cancer in the top 3 predictions?
        top3_indices = np.argsort(y_proba, axis=1)[:, -3:]
        top3_correct = sum(held_out_idx in row for row in top3_indices)
        top3_accuracy = top3_correct / len(y_test_enc)

        # Confidence for held-out class
        held_out_proba = y_proba[:, held_out_idx]
        mean_confidence = held_out_proba.mean()
        max_confidence = held_out_proba.max()

        # Overall max confidence (any class)
        max_pred_confidence = y_proba.max(axis=1).mean()

        # Misclassification pattern
        predicted_labels = le.inverse_transform(y_pred)
        misclass_counts = pd.Series(predicted_labels).value_counts().to_dict()

        print(f"  Top-1 accuracy: {top1_accuracy:.4f} ({top1_correct}/{len(y_test_enc)})")
        print(f"  Top-3 accuracy: {top3_accuracy:.4f} ({top3_correct}/{len(y_test_enc)})")
        print(f"  Mean confidence for {held_out}: {mean_confidence:.4f}")
        print(f"  Prediction distribution: {misclass_counts}")

        all_results[held_out] = {
            "n_test_samples": int(len(y_test_enc)),
            "top1_accuracy": float(top1_accuracy),
            "top1_correct": int(top1_correct),
            "top3_accuracy": float(top3_accuracy),
            "top3_correct": int(top3_correct),
            "mean_confidence_held_out": float(mean_confidence),
            "max_confidence_held_out": float(max_confidence),
            "mean_max_prediction_confidence": float(max_pred_confidence),
            "prediction_distribution": {str(k): int(v) for k, v in misclass_counts.items()},
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "per_sample_probabilities": y_proba.tolist(),
        }

    return all_results, cancer_types


def compute_summary_statistics(results, cancer_types):
    """Compute aggregate CUP performance metrics."""
    top1_accs = [results[ct]["top1_accuracy"] for ct in cancer_types]
    top3_accs = [results[ct]["top3_accuracy"] for ct in cancer_types]
    confidences = [results[ct]["mean_confidence_held_out"] for ct in cancer_types]

    # Weighted by number of test samples
    total_samples = sum(results[ct]["n_test_samples"] for ct in cancer_types)
    weighted_top1 = sum(
        results[ct]["top1_accuracy"] * results[ct]["n_test_samples"]
        for ct in cancer_types
    ) / total_samples
    weighted_top3 = sum(
        results[ct]["top3_accuracy"] * results[ct]["n_test_samples"]
        for ct in cancer_types
    ) / total_samples

    summary = {
        "n_cancer_types": len(cancer_types),
        "total_test_samples": total_samples,
        "macro_avg_top1_accuracy": float(np.mean(top1_accs)),
        "macro_std_top1_accuracy": float(np.std(top1_accs)),
        "macro_avg_top3_accuracy": float(np.mean(top3_accs)),
        "macro_std_top3_accuracy": float(np.std(top3_accs)),
        "weighted_avg_top1_accuracy": float(weighted_top1),
        "weighted_avg_top3_accuracy": float(weighted_top3),
        "mean_confidence": float(np.mean(confidences)),
        "min_top1_accuracy": float(np.min(top1_accs)),
        "max_top1_accuracy": float(np.max(top1_accs)),
        "per_cancer_top1": {ct: results[ct]["top1_accuracy"] for ct in cancer_types},
        "per_cancer_top3": {ct: results[ct]["top3_accuracy"] for ct in cancer_types},
    }

    print(f"\n{'='*70}")
    print("CUP SIMULATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Cancer types evaluated: {len(cancer_types)}")
    print(f"  Total test samples: {total_samples}")
    print(f"  Macro-average Top-1 accuracy: {np.mean(top1_accs):.4f} ± {np.std(top1_accs):.4f}")
    print(f"  Macro-average Top-3 accuracy: {np.mean(top3_accs):.4f} ± {np.std(top3_accs):.4f}")
    print(f"  Weighted Top-1 accuracy: {weighted_top1:.4f}")
    print(f"  Weighted Top-3 accuracy: {weighted_top3:.4f}")
    print(f"  Mean confidence: {np.mean(confidences):.4f}")
    print(f"\n  Per-cancer Top-1 | Top-3:")
    for ct in cancer_types:
        print(f"    {ct}: {results[ct]['top1_accuracy']:.4f} | {results[ct]['top3_accuracy']:.4f}")

    return summary


def plot_cup_results(results, cancer_types, output_dir):
    """Generate CUP simulation figures."""
    os.makedirs(output_dir, exist_ok=True)

    # Short names for plotting
    short_names = {ct: ct.replace("TCGA-", "") for ct in cancer_types}

    # --- Figure 1: Top-1 and Top-3 accuracy per cancer type ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(cancer_types))
    width = 0.35

    top1 = [results[ct]["top1_accuracy"] * 100 for ct in cancer_types]
    top3 = [results[ct]["top3_accuracy"] * 100 for ct in cancer_types]

    bars1 = ax.bar(x - width/2, top1, width, label="Top-1 Accuracy", color="#2196F3", edgecolor="white")
    bars2 = ax.bar(x + width/2, top3, width, label="Top-3 Accuracy", color="#4CAF50", edgecolor="white")

    ax.set_xlabel("Held-Out Cancer Type", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("CUP Simulation: Leave-One-Cancer-Out Classification Accuracy", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([short_names[ct] for ct in cancer_types], fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11)
    ax.axhline(y=90, color="gray", linestyle="--", alpha=0.5, label="Clinical threshold (90%)")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "cup_accuracy_per_cancer.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "cup_accuracy_per_cancer.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: cup_accuracy_per_cancer.png/pdf")

    # --- Figure 2: Misclassification heatmap ---
    n = len(cancer_types)
    misclass_matrix = np.zeros((n, n))

    for i, held_out in enumerate(cancer_types):
        dist = results[held_out]["prediction_distribution"]
        total = results[held_out]["n_test_samples"]
        for ct_pred, count in dist.items():
            j = cancer_types.index(ct_pred)
            misclass_matrix[i, j] = count / total * 100

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        misclass_matrix,
        annot=True,
        fmt=".1f",
        xticklabels=[short_names[ct] for ct in cancer_types],
        yticklabels=[short_names[ct] for ct in cancer_types],
        cmap="YlOrRd",
        vmin=0,
        vmax=100,
        ax=ax,
        cbar_kws={"label": "% of predictions"},
    )
    ax.set_xlabel("Predicted Cancer Type", fontsize=12)
    ax.set_ylabel("Held-Out (True) Cancer Type", fontsize=12)
    ax.set_title("CUP Simulation: Prediction Distribution When Cancer Type is Held Out", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "cup_misclassification_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "cup_misclassification_heatmap.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: cup_misclassification_heatmap.png/pdf")

    # --- Figure 3: Confidence distribution ---
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, held_out in enumerate(cancer_types):
        proba_matrix = np.array(results[held_out]["per_sample_probabilities"])
        held_out_idx = cancer_types.index(held_out)
        held_out_proba = proba_matrix[:, held_out_idx]

        ax = axes[i]
        ax.hist(held_out_proba, bins=20, color="#FF9800", edgecolor="white", alpha=0.85)
        ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.7)
        ax.set_title(short_names[held_out], fontsize=11, fontweight="bold")
        ax.set_xlabel("P(true type)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_xlim(0, 1)

    fig.suptitle("CUP Simulation: Prediction Confidence for Held-Out Cancer Type", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "cup_confidence_distributions.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "cup_confidence_distributions.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: cup_confidence_distributions.png/pdf")


def main():
    print("=" * 70)
    print("CUP SIMULATION: Leave-One-Cancer-Out Classification")
    print("=" * 70)

    # Load data
    feature_df, labels = load_integrated_features()

    # Run CUP simulation
    results, cancer_types = run_cup_simulation(feature_df, labels)

    # Compute summary
    summary = compute_summary_statistics(results, cancer_types)

    # Generate figures
    print("\nGenerating figures...")
    plot_cup_results(results, cancer_types, OUTPUT_DIR)

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save per-cancer results (without large probability arrays for JSON)
    results_save = {}
    for ct, r in results.items():
        results_save[ct] = {k: v for k, v in r.items() if k != "per_sample_probabilities"}
    with open(OUTPUT_DIR / "cup_per_cancer_results.json", "w") as f:
        json.dump(results_save, f, indent=2, default=str)

    # Save summary
    with open(OUTPUT_DIR / "cup_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save full results with probabilities (pickle for size)
    with open(OUTPUT_DIR / "cup_full_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print(f"\nAll results saved to {OUTPUT_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
