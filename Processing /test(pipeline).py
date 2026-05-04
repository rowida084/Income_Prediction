import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, f1_score, recall_score

# =========================
# Load Test Data
# =========================
test_data = pd.read_csv("test_data.csv")

# فصل features و target
X_test = test_data.drop("Income", axis=1)
y_test = test_data["Income"]

# =========================
# Load Trained Models
# =========================
models = {
    "Logistic Regression": joblib.load("Logistic Regression.pkl"),
    "KNN": joblib.load("KNN.pkl"),
    "Decision Tree": joblib.load("Decision Tree.pkl"),
    "Random Forest": joblib.load("Random Forest.pkl"),
    "SVM": joblib.load("SVM.pkl")
}

# =========================
# Evaluation
# =========================
results = []

for name, model in models.items():

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    recall = recall_score(y_test, preds)

    results.append([name, acc, f1, recall])

# =========================
# Results Table
# =========================
results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "F1-score", "Recall"]
)

print("\n🔥 Model Evaluation Results:\n")
print(results_df)
