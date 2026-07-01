#!/usr/bin/env python3
"""Study 1 Step 4b: Benchmark 12 architectures (manuscript Table 1 family)."""

import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy import stats
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.svm import SVC

import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import N_CV_FOLDS, N_INDEPENDENT_RUNS, OUTPUT_DIR, RANDOM_STATE, SMOTE_K_NEIGHBORS


@dataclass
class ModelSpec:
    name: str
    params_approx: str
    use_smote: bool
    builder: Callable[[int, int], Any]


def _smote_pipe(clf, seed: int, n_classes: int) -> ImbPipeline:
    return ImbPipeline([
        ("scaler", RobustScaler()),
        ("smote", SMOTE(
            random_state=seed,
            k_neighbors=SMOTE_K_NEIGHBORS,
            sampling_strategy="auto",
        )),
        ("clf", clf),
    ])


def _plain_pipe(clf) -> ImbPipeline:
    return ImbPipeline([
        ("scaler", RobustScaler()),
        ("clf", clf),
    ])


def build_specs(n_classes: int) -> list[ModelSpec]:
    return [
        ModelSpec("LightGBM + SMOTE", "3.2M", True, lambda s, _: lgb.LGBMClassifier(
            objective="multiclass", num_class=n_classes, n_estimators=300,
            learning_rate=0.05, num_leaves=31, max_depth=6, random_state=s, verbosity=-1,
        )),
        ModelSpec("Grad. Boosting + SMOTE", "2.8M", True, lambda s, _: GradientBoostingClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05, random_state=s,
        )),
        ModelSpec("XGBoost + SMOTE", "3.1M", True, lambda s, _: xgb.XGBClassifier(
            objective="multi:softprob", num_class=n_classes, n_estimators=300,
            max_depth=6, learning_rate=0.05, random_state=s, verbosity=0,
        )),
        ModelSpec("Random Forest", "2.5M", True, lambda s, _: RandomForestClassifier(
            n_estimators=300, max_depth=15, random_state=s, n_jobs=-1,
        )),
        ModelSpec("Grad. Boosting (no SMOTE)", "2.8M", False, lambda s, _: GradientBoostingClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05, random_state=s,
        )),
        ModelSpec("SVM (RBF)", "1.2K", True, lambda s, _: SVC(kernel="rbf", probability=False, random_state=s)),
        ModelSpec("Logistic Regression", "880", True, lambda s, _: LogisticRegression(
            max_iter=2000, random_state=s, multi_class="multinomial",
        )),
        ModelSpec("Deep Neural Network", "8M", True, lambda s, _: MLPClassifier(
            hidden_layer_sizes=(512, 256, 128, 64), max_iter=400, random_state=s,
        )),
        ModelSpec("Tabular MLP (TabTransformer proxy)", "12M", True, lambda s, _: MLPClassifier(
            hidden_layer_sizes=(768, 512, 256, 128, 64), max_iter=500, random_state=s,
        )),
        ModelSpec("Multi-layer MLP (Transformer proxy)", "58M", True, lambda s, _: MLPClassifier(
            hidden_layer_sizes=(1024, 1024, 512, 512, 256, 128), max_iter=600, random_state=s,
        )),
        ModelSpec("Stacking Ensemble", "3.5M", True, lambda s, _: StackingClassifier(
            estimators=[
                ("lgb", lgb.LGBMClassifier(
                    objective="multiclass", num_class=n_classes, n_estimators=150,
                    random_state=s, verbosity=-1,
                )),
                ("xgb", xgb.XGBClassifier(
                    objective="multi:softprob", num_class=n_classes, n_estimators=150,
                    random_state=s, verbosity=0,
                )),
                ("rf", RandomForestClassifier(n_estimators=150, random_state=s, n_jobs=-1)),
            ],
            final_estimator=LogisticRegression(max_iter=1000, random_state=s),
            cv=3,
        )),
        ModelSpec("LightGBM (no SMOTE)", "3.2M", False, lambda s, _: lgb.LGBMClassifier(
            objective="multiclass", num_class=n_classes, n_estimators=300,
            learning_rate=0.05, num_leaves=31, max_depth=6, random_state=s, verbosity=-1,
        )),
    ]


def cv_score(spec: ModelSpec, X: np.ndarray, y_enc: np.ndarray, seed: int) -> float:
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=seed)
    scores = []
    n_classes = len(np.unique(y_enc))
    for train_idx, val_idx in skf.split(X, y_enc):
        clf = spec.builder(seed, n_classes)
        pipe = _smote_pipe(clf, seed, n_classes) if spec.use_smote else _plain_pipe(clf)
        pipe.fit(X[train_idx], y_enc[train_idx])
        pred = pipe.predict(X[val_idx])
        scores.append(balanced_accuracy_score(y_enc[val_idx], pred))
    return float(np.mean(scores))


def main():
    print("=" * 60)
    print("Study 1 Step 4b: 12-architecture benchmark")
    print("=" * 60)

    feat_path = OUTPUT_DIR / "features_110.pkl"
    label_path = OUTPUT_DIR / "labels.csv"
    if not feat_path.exists():
        print("ERROR: run step3 first.")
        sys.exit(1)

    X_df = pd.read_pickle(feat_path)
    y_series = pd.read_csv(label_path, index_col=0).squeeze()
    if isinstance(y_series, pd.DataFrame):
        y_series = y_series.iloc[:, 0]

    le = LabelEncoder()
    y_enc = le.fit_transform(y_series.values)
    X = X_df.values.astype(float)
    n_classes = len(le.classes_)

    specs = build_specs(n_classes)
    assert len(specs) == 12

    table = []
    complexities = []
    accuracies = []

    for spec in specs:
        run_scores = []
        for run in range(N_INDEPENDENT_RUNS):
            seed = RANDOM_STATE + run
            try:
                score = cv_score(spec, X, y_enc, seed)
            except Exception as exc:
                print(f"  {spec.name} run {run + 1} failed: {exc}")
                score = float("nan")
            run_scores.append(score)
        valid = [s for s in run_scores if not np.isnan(s)]
        mean = float(np.mean(valid)) if valid else float("nan")
        std = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
        print(f"  {spec.name}: {mean:.1%} ± {std:.1%}")

        # Parse param count from label (e.g. "3.2M" -> 3.2e6)
        p = spec.params_approx.replace("M", "").replace("K", "")
        if "K" in spec.params_approx:
            comp = float(p) * 1e3
        else:
            comp = float(p) * 1e6

        table.append({
            "model": spec.name,
            "params_label": spec.params_approx,
            "params_numeric": comp,
            "use_smote": spec.use_smote,
            "cv_runs": run_scores,
            "cv_mean_balanced_accuracy": mean,
            "cv_std_balanced_accuracy": std,
            "cv_mean_pct": round(mean * 100, 1) if not np.isnan(mean) else None,
            "cv_std_pct": round(std * 100, 1),
        })
        if not np.isnan(mean):
            complexities.append(comp)
            accuracies.append(mean)

    # Complexity–accuracy correlation
    if len(complexities) >= 3:
        r, p = stats.pearsonr(complexities, accuracies)
        correlation = {"r_squared": round(r ** 2, 3), "pearson_r": round(r, 3), "p_value": float(p)}
    else:
        correlation = None

    out = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_folds": N_CV_FOLDS,
        "n_independent_runs": N_INDEPENDENT_RUNS,
        "models": table,
        "complexity_accuracy_correlation": correlation,
        "note": "TabTransformer/Multi-modal Transformer rows use MLP proxies at matched depth.",
    }

    with open(OUTPUT_DIR / "architecture_benchmark.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved -> {OUTPUT_DIR / 'architecture_benchmark.json'}")
    if correlation:
        print(f"Complexity vs accuracy: R²={correlation['r_squared']}, p={correlation['p_value']:.4f}")


if __name__ == "__main__":
    main()
