# Decision Tree Model 

# import libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# load data
train = pd.read_csv("processed_train_hott_data.csv")
test = pd.read_csv("processed_test_hott_data.csv")

# split features and target
X_train = train.drop("Income", axis=1)
y_train = train["Income"]

X_test = test.drop("Income", axis=1)
y_test = test["Income"]

# try different parameters 
best_model = None
best_acc = 0
best_params = {}

for depth in [8, 10, 12, 15]:
    for leaf in [1, 3, 5, 10]:
        for split in [2, 5, 10]:

            model = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_leaf=leaf,
                min_samples_split=split,
                random_state=42
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
                    "min_samples_split": split
                }

print("\nBest Parameters:", best_params)

# final model
y_pred = best_model.predict(X_test)

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
    best_model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("\nTop 10 Important Features:\n")
print(importance.head(10))

# save model
joblib.dump({"model": model, "columns": list(X_train.columns)}, "deci_model_rf.pkl")
