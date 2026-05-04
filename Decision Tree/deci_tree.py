# Decision Tree Model

# import libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score, recall_score, confusion_matrix, classification_report, f1_score
import joblib

# load data
train = pd.read_csv("processed_train_hott_data.csv")
test = pd.read_csv("processed_test_hott_data.csv")

# split features and target
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



# try different parameters
best_model = None
best_score  = 0
best_params = {}

for depth in [ 10, 15, 20]:
    for leaf in [1, 3, 5]:
        for split in [2, 5, 10]:

            model = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_leaf=leaf,
                min_samples_split=split,
                criterion='entropy',
                random_state=42
            )


            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred)

            

            if f1 > best_score :
                best_score  = f1
                best_model = model
                best_params = {
                    "max_depth": depth,
                    "min_samples_leaf": leaf,
                    "min_samples_split": split
                }

print("\nBest Parameters:", best_params)



# =========================
# Final Training (Train + Val)
# =========================
final_model = DecisionTreeClassifier(**best_params, criterion='entropy', random_state=42)
final_model.fit(X, y)

# =========================
# Test (one time only)
# =========================
X_test = test.drop("Income", axis=1)
y_test = test["Income"]

y_pred = final_model.predict(X_test)



test_precision = precision_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)



# accuracy
acc = accuracy_score(y_test, y_pred)
print("\nFinal Accuracy:", acc)
print("Final F1-score:", test_f1)
print("Final Precision:", test_precision)
print("Final Recall:", test_recall)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Feature Importance
importance = pd.Series(
    best_model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("\nTop 10 Important Features:\n")
print(importance.head(10))

# save model
# joblib.dump({"model": model, "columns": list(X_train.columns)}, "deci_model_rf.pkl")
