import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import joblib
import argparse
import yaml

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train cancer classification model')
parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
args = parser.parse_args()

# Load configuration
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# Load and prepare data
data = pd.read_csv("data/processed/dataset.csv")
X = data.drop("label", axis=1)
y = data["label"]

# Train model with configuration
k_folds = config['train']['k_folds']
random_state = config['train']['random_state']
n_estimators = config['model']['n_estimators']
max_depth = config['model']['max_depth']

# K-Fold Cross-Validation
cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
cross_val_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

print(f"Cross-Validation Scores: {cross_val_scores}")
print(f"Mean CV Accuracy: {cross_val_scores.mean():.3f}, Std: {cross_val_scores.std():.3f}")

# Train final model
model.fit(X, y)

# Save model
joblib.dump(model, "models/latest.pkl")
print(f"Model trained and saved to models/latest.pkl")
print(f"Final model accuracy on full dataset: {model.score(X, y):.3f}")
