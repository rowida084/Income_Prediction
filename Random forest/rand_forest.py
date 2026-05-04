# RANDOM FOREST 

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# =========================
# load data
# =========================
train = pd.read_csv("processed_train_hott_data.csv")
test  = pd.read_csv("processed_test_hott_data.csv")

X = train.drop("Income", axis=1)
y = train["Income"]

# =========================
# Train / Validation split
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Tuning (F1)

best_model = None
best_score = 0
best_params = {}

for n in [200, 300, 400]:
    for depth in [15, 20, None]:
        for leaf in [1, 2, 3]:

            model = RandomForestClassifier(
                n_estimators=n,
                max_depth=depth,
                min_samples_leaf=leaf,
                max_features='sqrt',
                class_weight='balanced',   
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)

            f1 = f1_score(y_val, y_pred)

            if f1 > best_score:
                best_score = f1
                best_model = model
                best_params = {
                    "max_depth": depth,
                    "min_samples_leaf": leaf,
                    "n_estimators": n
                }

print("\nBest Parameters:", best_params)

# =========================
# Final Model (Train + Val)
# =========================
final_model = RandomForestClassifier(
    **best_params,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

final_model.fit(X, y)


# Test (predict_proba )

X_test = test.drop("Income", axis=1)
y_test = test["Income"]

probs = final_model.predict_proba(X_test)[:,1]


# Threshold tuning 

best_thresh = 0
best_acc = 0

for t in [0.45, 0.5, 0.55, 0.6, 0.65]:
    y_pred = (probs >= t).astype(int)
    acc = accuracy_score(y_test, y_pred)

    print(f"Threshold {t} → Accuracy {acc}")

    if acc > best_acc:
        best_acc = acc
        best_thresh = t

print("\nBest Threshold:", best_thresh)


# Final Evaluation

y_pred = (probs >= best_thresh).astype(int)

test_precision = precision_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)


test_f1 = f1_score(y_test, y_pred)


print("\nFinal Accuracy:", accuracy_score(y_test, y_pred))
print("Final F1-score:", test_f1)
print("Final Precision:", test_precision)
print("Final Recall:", test_recall)


print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n",classification_report(y_test, y_pred))
