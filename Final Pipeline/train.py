import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from preprocessing import FeatureEngineering
from models import train_model

# =========================
#  Load Data
# =========================
data = pd.read_csv("train_cleaned_v1.csv")

data['Income'] = data['Income'].astype(str).str.strip().str.replace('.', '', regex=False)
data['Income'] = data['Income'].map({'>50K': 1, '<=50K': 0})

data = data.drop_duplicates()

# data = data.dropna(subset=['Income'])

X = data.drop("Income", axis=1)
y = data["Income"]

# =========================
#  Split
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
#  Columns
# =========================
num_cols = [
    "age", "education-num", "capital-gain",
    "capital-loss", "hours-per-week",
    "net_capital", "work_status"
]

cat_cols = [
    "workclass", "marital-status",
    "occupation", "relationship", "native-country"
]

# =========================
#  Pipelines
# =========================
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = {
    "features": FeatureEngineering(),
    "transform": ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])
  


 # not true ik consider it a placeholer   
for name, config in models_config.items():

    print(f"\n Training {name}")

    trainer = config["trainer"]
    params  = config["params"]

    model = trainer(X_train, y_train, X_val, y_val, params)

    # handling KNN scaler
    if isinstance(model, tuple):
        model, scaler = model
        X_test_used = scaler.transform(X_test)
    else:
        X_test_used = X_test

    y_pred = model.predict(X_test_used)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    
