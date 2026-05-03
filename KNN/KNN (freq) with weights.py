
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score
from sklearn.neighbors import KNeighborsClassifier



# =========================
# KNN FROM SCRATCH (IMPROVED)
# =========================
class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
     distances = np.linalg.norm(self.X_train - x, axis=1)

     k_indices = np.argsort(distances)[:self.k]
     k_labels = self.y_train[k_indices]
     k_distances = distances[k_indices]

     weights = 1 / (k_distances + 1e-5)

     label_score = {}

     for label, w in zip(k_labels, weights):
        label_score[label] = label_score.get(label, 0) + w

     return max(label_score, key=label_score.get)
# =========================
# LOAD DATA
# =========================
train_df = pd.read_csv("processed_train_freq_data.csv")
test_df = pd.read_csv("processed_test_freq_data.csv")

X = train_df.drop("Income", axis=1)
y = train_df["Income"]
X = X.fillna(X.median())

X_test = test_df.drop("Income", axis=1)
y_test = test_df["Income"]

corr = X.corrwith(y).abs()
selected_features = corr[corr > 0.01].index
X = X[selected_features]


# =========================
# TRAIN / VALID SPLIT
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
# =========================
# SCALING (VERY IMPORTANT FOR KNN)
# =========================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# =========================
# FIND BEST K
# =========================
best_k = None
best_acc = 0
best_f1 =0
best_score = 0




for k in range(1, 60, 2):# the best accuracy 83.49 with k = 37

    model = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring='f1_weighted'
    )

    acc_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring='accuracy'
    )

    mean_f1 = scores.mean()
    mean_acc = acc_scores.mean()

    print(f"k = {k} → F1 = {mean_f1:.4f} | Accuracy = {mean_acc:.4f}")

    if mean_f1 > best_score:
        best_score = mean_f1
        best_k = k
        best_acc = mean_acc

print(f"\nBest k: {best_k}")
print(f"Best CV F1: {best_score:.4f}")
print(f"Best CV Accuracy: {best_acc:.4f}")
# =========================
# FINAL MODEL
# =========================
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train, y_train)

test_pred = final_model.predict(X_test)

test_acc = accuracy_score(y_test, test_pred)
test_f1 = f1_score(y_test, test_pred, average='weighted')

print("\nFinal Test Accuracy:", test_acc)
print("Final Test F1-score:", test_f1)
