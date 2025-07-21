import pandas as pd
import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Evaluate cancer classification model')
parser.add_argument('--model', type=str, default='models/latest.pkl', help='Path to model file')
args = parser.parse_args()

# Load model and data
model = joblib.load(args.model)
data = pd.read_csv("data/processed/dataset.csv")
X = data.drop("label", axis=1)
y = data["label"]

# Evaluate with cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
print(f"Cross-Validation Results: {cv_results}")
print(f"Mean CV Accuracy: {cv_results.mean():.3f}, Std: {cv_results.std():.3f}")

# Make predictions and evaluate
preds = model.predict(X)
print(f"Model evaluation for: {args.model}")
print(classification_report(y, preds))
