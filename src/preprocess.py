import pandas as pd
import os

# Simulate preprocessing
data = pd.read_csv("data/raw/sample.csv")
data = data.dropna()
data.to_csv("data/processed/dataset.csv", index=False)
