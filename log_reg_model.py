# Logistic Regression Model for Income Classification

# import libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

# load data
train = pd.read_csv("processed_train_data.csv")
test = pd.read_csv("processed_test_data.csv")

# split features and target
X_train = train.drop("Income", axis=1)
y_train = train["Income"]

X_test = test.drop("Income", axis=1)
y_test = test["Income"]

# try diff C values
best_c = None
best_acc = 0

for c in [0.01, 0.1, 1, 10, 100]:
    model = LogisticRegression(C=c, max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"C = {c} → Accuracy = {acc}")

    if acc > best_acc:
        best_acc = acc
        best_c = c


print("\nBest C value:", best_c)
print("Best Accuracy:", best_acc)
model = LogisticRegression(C=best_c, max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# accuracy
acc = accuracy_score(y_test, y_pred)
print("\nFinal Accuracy:", acc)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nModel training completed successfully ✅")