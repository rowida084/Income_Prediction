import pandas as pd
import numpy as np
import joblib
 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
 
from preprocessing import FeatureEngineering
from models import models_config
 
# ============================================================
# STEP 1 — Load Data
# ============================================================
train_raw = pd.read_csv("train_data.csv")
test_raw  = pd.read_csv("test_data.csv")
 
# fix target
for df in [train_raw, test_raw]:
    df.rename(columns={"Income ": "Income"}, inplace=True)
    df['Income'] = df['Income'].astype(str).str.strip().str.replace('.', '', regex=False)
    df['Income'] = df['Income'].map({'>50K': 1, '<=50K': 0})
 
train_raw = train_raw.drop_duplicates()
 
X_raw    = train_raw.drop('Income', axis=1)
y        = train_raw['Income']
X_test   = test_raw.drop('Income', axis=1)
y_test   = test_raw['Income']
 
# ============================================================
# STEP 2 — Train / Val Split
# ============================================================
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_raw, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
 
# ============================================================
# STEP 3 — Columns بعد الـ FeatureEngineering
# ============================================================
num_cols = [
    "age", "education-num", "capital-gain", "capital-loss",
    "hours-per-week", "net_capital", "work_status"
]
 
cat_cols = [
    "workclass", "marital-status", "occupation",
    "relationship", "native-country"
]
 
# ============================================================
# STEP 4 — Preprocessor Pipeline
# ============================================================
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])
 
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
 
column_transformer = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols),
    # has_gain و sex بيفضلوا زي ما هم (binary)
], remainder='passthrough')
 
full_preprocessor = Pipeline([
    ("feature_eng", FeatureEngineering()),
    ("transform",   column_transformer)
])
 
# ============================================================
# STEP 5 — Fit Preprocessor على الـ Train
# ============================================================
X_train_proc = full_preprocessor.fit_transform(X_train_raw, y_train)
X_val_proc   = full_preprocessor.transform(X_val_raw)
X_test_proc  = full_preprocessor.transform(X_test)
 
print(f"Train shape after preprocessing: {X_train_proc.shape}")
print(f"Val   shape after preprocessing: {X_val_proc.shape}")
print(f"Test  shape after preprocessing: {X_test_proc.shape}")
 
# ============================================================
# STEP 6 — Train All Models
# ============================================================
trained_models = {}
 
for name, config in models_config.items():
    print(f"\n{'='*50}")
    print(f"Training: {name}")
    print('='*50)
 
    trainer = config["trainer"]
    params  = config["params"]
 
    model = trainer(X_train_proc, y_train, X_val_proc, y_val, params)
 
    # Final training على كل الداتا (train + val)
    X_full_proc = full_preprocessor.transform(X_raw)
    model.fit(X_full_proc, y)
 
    # Evaluate على الـ test
    y_pred = model.predict(X_test_proc)
 
    print(f"\nTest Results — {name}:")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  F1 Score : {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print(classification_report(y_test, y_pred))
 
    # Save pipeline (preprocessor + model)
    full_pipeline = Pipeline([
        ("preprocessor", full_preprocessor),
        ("model",        model)
    ])
    filename = f"pipeline_{name.lower()}.pkl"
    joblib.dump(full_pipeline, filename)
    print(f"  Saved → {filename}")
 
    trained_models[name] = full_pipeline
 
print("\nAll models trained and saved ✅")
