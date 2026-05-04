import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, f1_score

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
# Preprocessing
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
# Models
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC()
}

# =========================
# Train + Evaluate
# =========================
results = []

for name, model in models.items():

    pipeline = Pipeline([
        ("features", FeatureEngineering()),
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X, y)

    preds = pipeline.predict(X)

    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)

    results.append([name, acc, f1])

    
    joblib.dump(pipeline, f"{name}.pkl")

# =========================
# Results
# =========================
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1-score"])
print(results_df)
