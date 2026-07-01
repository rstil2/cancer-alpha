"""Shared demo logic for Streamlit and native desktop app."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

FEATURE_NAMES = (
    [f"methylation_{i}" for i in range(20)]
    + [f"mutation_{i}" for i in range(25)]
    + [f"cn_alteration_{i}" for i in range(20)]
    + [f"fragmentomics_{i}" for i in range(15)]
    + [f"clinical_{i}" for i in range(10)]
    + [f"icgc_argo_{i}" for i in range(20)]
)

CANCER_TYPES = ["BRCA", "LUAD", "COAD", "PRAD", "STAD", "KIRC", "HNSC", "LIHC"]

MODEL_FILES = {
    "Random Forest": "multimodal_real_tcga_random_forest.pkl",
    "Logistic Regression": "multimodal_real_tcga_logistic_regression.pkl",
}

MODALITY_SLICES = [
    ("methylation", 0, 20),
    ("mutation", 20, 45),
    ("cna", 45, 65),
    ("fragmentomics", 65, 80),
    ("clinical", 80, 90),
    ("icgc", 90, 110),
]


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent


class DemoEngine:
    def __init__(self) -> None:
        self.models_dir = app_dir() / "models"
        self.models: dict = {}
        self.scalers: dict = {}
        self.feature_selector = None
        self._load()

    def _load(self) -> None:
        for name, filename in MODEL_FILES.items():
            path = self.models_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing model file: {path}")
            self.models[name] = joblib.load(path)

        scalers_path = self.models_dir / "scalers.pkl"
        if not scalers_path.exists():
            raise FileNotFoundError(f"Missing scalers: {scalers_path}")
        scalers_data = joblib.load(scalers_path)
        self.scalers = scalers_data if isinstance(scalers_data, dict) else {"main": scalers_data}

        selector_path = self.models_dir / "feature_selector.pkl"
        if selector_path.exists():
            self.feature_selector = joblib.load(selector_path)

    @staticmethod
    def generate_sample(kind: str = "cancer") -> np.ndarray:
        np.random.seed(42 if kind == "cancer" else 24)
        if kind == "cancer":
            parts = [
                np.random.normal(0.5, 0.1, 20),
                np.random.poisson(5, 25),
                np.random.normal(10, 2, 20),
                np.random.exponential(150, 15),
                np.random.normal(0.5, 0.1, 10),
                np.random.gamma(2, 0.5, 20),
            ]
        else:
            parts = [
                np.random.normal(-0.2, 0.03, 20),
                np.random.poisson(0.5, 25),
                np.random.normal(0, 0.5, 20),
                np.random.exponential(200, 15),
                np.random.normal(-0.3, 0.03, 10),
                np.random.gamma(0.8, 0.2, 20),
            ]
        return np.concatenate(parts)

    def preprocess(self, input_data: np.ndarray, model_name: str) -> np.ndarray:
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)

        scaled_parts = []
        for key, start, end in MODALITY_SLICES:
            block = input_data[:, start:end]
            if key in self.scalers:
                scaled_parts.append(self.scalers[key].transform(block))
            else:
                scaled_parts.append(block)
        scaled = np.hstack(scaled_parts)

        if model_name == "Logistic Regression" and self.feature_selector is not None:
            scaled = self.feature_selector.transform(scaled)
        return scaled

    def predict(self, model_name: str, input_data: np.ndarray) -> dict:
        model = self.models[model_name]
        processed = self.preprocess(input_data, model_name)
        prediction = int(model.predict(processed)[0])
        probabilities = model.predict_proba(processed)[0]
        prob_sum = probabilities.sum()
        if prob_sum > 0 and abs(prob_sum - 1.0) > 0.01:
            probabilities = probabilities / prob_sum
        return {
            "predicted_cancer_type": CANCER_TYPES[prediction],
            "confidence_score": float(max(probabilities)),
            "class_probabilities": probabilities.tolist(),
            "cancer_types": CANCER_TYPES,
        }
