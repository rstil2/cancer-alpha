import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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
test_size = config['train']['test_size']
random_state = config['train']['random_state']
n_estimators = config['model']['n_estimators']
max_depth = config['model']['max_depth']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/latest.pkl")
print(f"Model trained and saved to models/latest.pkl")
print(f"Training accuracy: {model.score(X_train, y_train):.3f}")
print(f"Test accuracy: {model.score(X_test, y_test):.3f}")
