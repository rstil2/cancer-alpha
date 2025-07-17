import pandas as pd
import joblib
from sklearn.metrics import classification_report

model = joblib.load("models/latest.pkl")
data = pd.read_csv("data/processed/dataset.csv")
X = data.drop("label", axis=1)
y = data["label"]

preds = model.predict(X)
print(classification_report(y, preds))
