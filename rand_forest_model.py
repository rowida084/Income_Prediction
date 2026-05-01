# RANDOM FOREST MODEL
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# load data
train = pd.read_csv("processed_train_hott_data.csv")
test  = pd.read_csv("processed_test_hott_data.csv")

# split features and target
X_train = train.drop("Income", axis=1)
y_train = train["Income"]
X_test  = test.drop("Income", axis=1)
y_test  = test["Income"]

# train model

best_model = None
best_acc = 0
best_params = {}


for n in [50, 100, 150]:
    for depth in [10, 15, 20]:
        for leaf in [1, 2, 4]:

            model = RandomForestClassifier(
                n_estimators=n,
                max_depth=depth,
                min_samples_leaf=leaf,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_params = {
                    "max_depth": depth,
                    "min_samples_leaf": leaf,
                    "n_estimators": n
                }


print("\nBest Parameters:", best_params)
# accuracy
acc = accuracy_score(y_test, y_pred)
print("\nFinal Accuracy:", acc)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Feature Importance
importance = pd.Series(
    model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("\nTop 10 Important Features:\n")
print(importance.head(10))

# save model
joblib.dump({"model": model, "columns": list(X_train.columns)}, "model_rf.pkl")
