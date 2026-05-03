import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

train = pd.read_csv("processed_train_freq_data.csv")
test = pd.read_csv("processed_test_freq_data.csv")

X_train = train.drop("Income", axis=1)
y_train = train["Income"]

X_test = test.drop("Income", axis=1)
y_test = test["Income"]

best_c = None
best_score = 0

# ======================
# TUNING C
# ======================
for c in [0.01, 0.1, 1, 10, 100]:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=c, max_iter=1000, class_weight='balanced'))
    ])
    
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
    mean_score = scores.mean()

    print(f"C = {c} → F1 = {mean_score}")

    if mean_score > best_score:
        best_score = mean_score
        best_c = c

print("\nBest C:", best_c)

# ======================
# FINAL MODEL
# ======================
final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=best_c, max_iter=1000, class_weight='balanced'))
])

final_model.fit(X_train, y_train)

# ======================
# PROBABILITIES
# ======================
probs = final_model.predict_proba(X_test)[:, 1]

# ======================
# THRESHOLD TUNING
# ======================
precision, recall, thresholds = precision_recall_curve(y_test, probs)

best_threshold = thresholds[(precision * recall).argmax()]
print("Best threshold:", best_threshold)

# ======================
# FINAL PREDICTION
# ======================
#y_pred = (probs >= best_threshold).astype(int)
y_pred = (probs >= 0.65).astype(int)
# ======================
# EVALUATION
# ======================
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
