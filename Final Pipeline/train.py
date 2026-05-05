import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score, f1_score,recall_score, confusion_matrix

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
train = pd.read_csv("train_data.csv")
test  = pd.read_csv("test_data.csv")

for data in [train, test]:
    data.rename(columns={"Income ": "Income"}, inplace=True)
    data['Income'] = data['Income'].astype(str).str.strip().str.replace('.', '', regex=False)
    data['Income'] = data['Income'].map({'>50K': 1, '<=50K': 0})

train = train.drop_duplicates()

# data = data.dropna(subset=['Income'])

X = train.drop("Income", axis=1)
y = train["Income"]

X_test   = test.drop('Income', axis=1)
y_test   = test['Income']

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

preprocessor = Pipeline([
    ("features", FeatureEngineering()),
    ("transform", ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ], remainder='passthrough'))
])

# ============================================================
# Train All Models
# ============================================================   
trained_models = {}
for name, config in models_config.items():

    print(f"\n Training {name}")

    trainer = config["trainer"]
    params  = config["params"]

    model = trainer(params)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    # TRAIN
    pipeline.fit(X_train, y_train)
    
    
    # VALIDATION
    y_val_pred  = model.predict(X_val)



    # FINAL TRAIN (train + val)
    pipeline.fit(X, y)

    # TEST
    y_test_pred = pipeline.predict(X_test)

    
   
    print(f"\nTest Results — {name}:")
    print(f"  Accuracy : {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"  F1 Score : {f1_score(y_test, y_test_pred):.4f}")
    print(f"  Recall   : {recall_score(y_test, y_test_pred):.4f}")
    print(f"  \nConfusion Matrix   : {confusion_matrix(y_test, y_test_pred)}")
    
    print(classification_report(y_test, y_test_pred))


    # Save 

    filename = f"pipeline_{name.lower()}.pkl"
    joblib.dump(pipeline, filename)
 
    trained_models[name] = full_pipeline
print("\nAll models trained and saved ")
