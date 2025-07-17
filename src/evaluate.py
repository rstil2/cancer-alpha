import pandas as pd
import joblib
from sklearn.metrics import classification_report
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

# Make predictions and evaluate
preds = model.predict(X)
print(f"Model evaluation for: {args.model}")
print(classification_report(y, preds))
