import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from feature_engineering import FeatureEngineering

# =========================
# Load data
# =========================
data = pd.read_csv("train_cleaned_v1.csv")

data.replace(['?', ' ?', '? '], pd.NA, inplace=True)
data.rename(columns={"Income ": "Income"}, inplace=True)

data['Income'] = data['Income'].astype(str).str.strip().str.replace('.', '', regex=False)

data['Income'] = data['Income'].map({'>50K': 1, '<=50K': 0})

data = data.dropna(subset=['Income'])

X = data.drop("Income", axis=1)
y = data["Income"]

# =========================
# Columns
# =========================
num_cols = ["age", "education-num", "capital-gain",
            "capital-loss", "hours-per-week",
            "net_capital", "work_status", "has_gain"]

cat_cols = ["workclass", "marital-status", "occupation",
            "relationship", "native-country", "sex"]

# =========================
# Pipelines
# =========================
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# =========================
# Full pipeline
# =========================
pipeline = Pipeline([
    ("features", FeatureEngineering()),
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(n_estimators=200, random_state=42))
])

# =========================
# Train
# =========================
pipeline.fit(X, y)

joblib.dump(pipeline, "pipeline.pkl")

