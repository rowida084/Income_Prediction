import pandas as pd
import joblib

# لازم تستورد الكلاس عشان joblib يلقاه
from feature_engineering import FeatureEngineering

# =========================
# Load model
# =========================
pipeline = joblib.load("pipeline.pkl")

# =========================
# Load test data
# =========================
test_data = pd.read_csv("test_data.csv")

test_data.replace(['?', ' ?', '? '], pd.NA, inplace=True)

test_data = test_data.drop("Income", axis=1, errors="ignore")

# =========================
# Predict
# =========================
predictions = pipeline.predict(test_data)

# =========================
# Save
# =========================
pd.DataFrame({"prediction": predictions}).to_csv("submission.csv", index=False)

