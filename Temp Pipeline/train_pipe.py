import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix, classification_report)
import joblib

# ============================================================
# STEP 1 — Custom Preprocessor
# ============================================================
class IncomePreprocessor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        X = X.copy()

        # drop fnlwgt و education (education-num بيغنى عنها)
        X = X.drop(columns=['fnlwgt', 'education'], errors='ignore')

        # fix column name
        X = X.rename(columns={"Income ": "Income"})

        # replace ? with NaN
        X = X.replace(['?', ' ?', '? '], np.nan)
        X['Income'] = X['Income'].str.replace('.', '', regex=False)

        # strip spaces
        for col in X.select_dtypes(include='object').columns:
            X[col] = X[col].str.strip()

        # numeric cols
        numeric_cols = ["age", "education-num", "capital-gain",
                        "capital-loss", "hours-per-week"]
        for col in numeric_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        # save medians from train
        self.medians_ = {col: X[col].median() for col in numeric_cols}

        # save modes from train
        self.modes_ = {col: X[col].mode()[0]
                       for col in X.select_dtypes(include='object').columns}

        # age bounds
        Q1 = X['age'].quantile(0.25)
        Q3 = X['age'].quantile(0.75)
        IQR = Q3 - Q1
        self.age_lower_ = Q1 - 1.5 * IQR
        self.age_upper_ = Q3 + 1.5 * IQR

        # fit scaler على الـ numeric cols بعد الـ feature engineering
        X_transformed = self._transform_core(X)
        scale_cols = ["age", "education-num", "capital-gain",
                      "capital-loss", "hours-per-week", "work_status", "net_capital"]
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X_transformed[scale_cols])

        # save columns order
        self.feature_names_ = X_transformed.columns.tolist()

        return self

    def _transform_core(self, X):
        X = X.copy()

        # drop
        X = X.drop(columns=['fnlwgt', 'education'], errors='ignore')
        X = X.rename(columns={"Income ": "Income"})
        X = X.replace(['?', ' ?', '? '], np.nan)
        X['Income'] = X['Income'].str.replace('.', '', regex=False)
      
        for col in X.select_dtypes(include='object').columns:
            X[col] = X[col].str.strip()

        numeric_cols = ["age", "education-num", "capital-gain",
                        "capital-loss", "hours-per-week"]
        for col in numeric_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].fillna(self.medians_[col])

        for col in X.select_dtypes(include='object').columns:
            X[col] = X[col].fillna(self.modes_.get(col, X[col].mode()[0]))

        # feature engineering
        X['native-country'] = X['native-country'].apply(
            lambda x: 'USA' if x == 'United-States' else 'Other')

        X['net_capital'] = X['capital-gain'] - X['capital-loss']
        X['has_gain'] = (X['capital-gain'] > 0).astype(int)

        def work_status(h):
            if h < 35:   return 0
            elif h <= 45: return 1
            else:         return 2
        X['work_status'] = X['hours-per-week'].apply(work_status)

        # encoding
        X['sex'] = X['sex'].map({'Male': 1, 'Female': 0})

        # outliers
        X['age'] = X['age'].clip(self.age_lower_, self.age_upper_)
        X['capital-gain'] = np.log1p(X['capital-gain'])
        X['capital-loss'] = np.log1p(X['capital-loss'])

        # one-hot encoding
        categorical_cols = ["workclass", "marital-status",
                            "occupation", "relationship", "native-country"]
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        return X

    def transform(self, X):
        X = self._transform_core(X)

        # align columns مع الـ train
        X = X.reindex(columns=self.feature_names_, fill_value=0)

        # scaling
        scale_cols = ["age", "education-num", "capital-gain",
                      "capital-loss", "hours-per-week", "work_status", "net_capital"]
        X[scale_cols] = self.scaler_.transform(X[scale_cols])

        return X


# ============================================================
# STEP 2 — Load Raw Data
# ============================================================
train_raw = pd.read_csv("train_data.csv")
test_raw  = pd.read_csv("test_data.csv")

# fix target column name
train_raw = train_raw.rename(columns={"Income ": "Income"})
test_raw  = test_raw.rename(columns={"Income ": "Income"})

# clean target
for df in [train_raw, test_raw]:
    df['Income'] = df['Income'].astype(str).str.strip().str.replace('.', '', regex=False)

# encode target
train_raw['Income'] = train_raw['Income'].map({'>50K': 1, '<=50K': 0})
test_raw['Income']  = test_raw['Income'].map({'>50K': 1, '<=50K': 0})

X_raw        = train_raw.drop('Income', axis=1)
y            = train_raw['Income']
X_test_raw   = test_raw.drop('Income', axis=1)
y_test       = test_raw['Income']

# Train / Val split
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y)


# ============================================================
# STEP 3 — Pipelines
# ============================================================

# ── Decision Tree ──────────────────────────────────────────
dt_pipeline = Pipeline([
    ('preprocessor', IncomePreprocessor()),
    ('model', DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=3,
        min_samples_split=2,
        criterion='entropy',
        random_state=42
    ))
])

dt_pipeline.fit(X_raw, y)
y_pred_dt = dt_pipeline.predict(X_test_raw)

print("=" * 50)
print("DECISION TREE")
print("=" * 50)
print("Accuracy :", accuracy_score(y_test, y_pred_dt))
print("F1       :", f1_score(y_test, y_pred_dt, average='weighted'))
print("Precision:", precision_score(y_test, y_pred_dt, average='weighted'))
print("Recall   :", recall_score(y_test, y_pred_dt, average='weighted'))
print(classification_report(y_test, y_pred_dt))

joblib.dump(dt_pipeline, "pipeline_decision_tree.pkl")
print("Saved → pipeline_decision_tree.pkl\n")


# ── Logistic Regression ────────────────────────────────────
lr_pipeline = Pipeline([
    ('preprocessor', IncomePreprocessor()),
    ('model', LogisticRegression(
        C=1,
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ))
])

lr_pipeline.fit(X_raw, y)
y_pred_lr = lr_pipeline.predict(X_test_raw)

print("=" * 50)
print("LOGISTIC REGRESSION")
print("=" * 50)
print("Accuracy :", accuracy_score(y_test, y_pred_lr))
print("F1       :", f1_score(y_test, y_pred_lr, average='weighted'))
print("Precision:", precision_score(y_test, y_pred_lr, average='weighted'))
print("Recall   :", recall_score(y_test, y_pred_lr, average='weighted'))
print(classification_report(y_test, y_pred_lr))

joblib.dump(lr_pipeline, "pipeline_logistic_regression.pkl")
print("Saved → pipeline_logistic_regression.pkl\n")


# ── Random Forest ──────────────────────────────────────────
rf_pipeline = Pipeline([
    ('preprocessor', IncomePreprocessor()),
    ('model', RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

rf_pipeline.fit(X_raw, y)
y_pred_rf = rf_pipeline.predict(X_test_raw)

print("=" * 50)
print("RANDOM FOREST")
print("=" * 50)
print("Accuracy :", accuracy_score(y_test, y_pred_rf))
print("F1       :", f1_score(y_test, y_pred_rf, average='weighted'))
print("Precision:", precision_score(y_test, y_pred_rf, average='weighted'))
print("Recall   :", recall_score(y_test, y_pred_rf, average='weighted'))
print(classification_report(y_test, y_pred_rf))

joblib.dump(rf_pipeline, "pipeline_random_forest.pkl")
print("Saved → pipeline_random_forest.pkl\n")


# ============================================================
# STEP 4 — Predict on new sample (مثال)
# ============================================================
# لما تيجي تتنبأ على بيانات جديدة بتعمل كده:
#
# pipeline = joblib.load("pipeline_decision_tree.pkl")
# new_data = pd.DataFrame([{
#     'age': 35, 'workclass': 'Private', 'fnlwgt': 123456,
#     'education': 'Bachelors', 'education-num': 13,
#     'marital-status': 'Married-civ-spouse', 'occupation': 'Exec-managerial',
#     'relationship': 'Husband', 'race': 'White', 'sex': 'Male',
#     'capital-gain': 0, 'capital-loss': 0, 'hours-per-week': 40,
#     'native-country': 'United-States'
# }])
# prediction = pipeline.predict(new_data)
# print("Prediction:", ">50K" if prediction[0] == 1 else "<=50K")
