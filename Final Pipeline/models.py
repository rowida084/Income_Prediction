from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score


# ── Decision Tree ──────────────────────────────────────────
def train_decision_tree(X_train, y_train, X_val, y_val, params):

    best_model = None
    best_score = 0

    for depth in params["max_depth"]:
        for leaf in params["min_samples_leaf"]:
            for split in params["min_samples_split"]:

                model = DecisionTreeClassifier(
                    max_depth=depth,
                    min_samples_leaf=leaf,
                    min_samples_split=split,
                    criterion='entropy',
                    random_state=42
                )

                model.fit(X_train, y_train)
                pred = model.predict(X_val)

                f1 = f1_score(y_val, pred)

                if f1 > best_score:
                    best_score = f1
                    best_model = model

    print(f"  Best Val F1: {best_score:.4f}")
    return best_model


# ── Random Forest ──────────────────────────────────────────
def train_random_forest(X_train, y_train, X_val, y_val, params):

    best_model = None
    best_score = 0

    for n in params["n_estimators"]:
        for depth in params["max_depth"]:
            for leaf in params["min_samples_leaf"]:

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
                pred = model.predict(X_val)

                f1 = f1_score(y_val, pred)

                if f1 > best_score:
                    best_score = f1
                    best_model = model
    print(f"  Best Val F1: {best_score:.4f}")
    return best_model


 # ── Logistic Regression ────────────────────────────────────
def train_logistic(X_train, y_train, X_val, y_val, params):
    best_model = None
    best_score = 0

    for c in params["C"]:
        model = LogisticRegression(
            C=c,
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_train, y_train)
        f1 = f1_score(y_val, model.predict(X_val))
        if f1 > best_score:
            best_score = f1
            best_model = model

    print(f"  Best Val F1: {best_score:.4f}")
    return best_model



# ── SVM ────────────────────────────────────────────────────
def train_svm(X_train, y_train, X_val, y_val, params):

    model = SVC(
        kernel='rbf',
        C=params["C"],
        gamma=params["gamma"],
        class_weight={0:1, 1:2},
        probability=True,
        random_state=42
    )

    model.fit(X_train, y_train)

    f1 = f1_score(y_val, model.predict(X_val))
    print(f"  Val F1: {f1:.4f}")

    return model


# ── KNN ────────────────────────────────────
def train_knn(X_train, y_train, X_val, y_val, params):
    best_model = None
    best_score = 0

    for k in params["n_neighbors"]:
        for w in params["weights"]:  
            model = KNeighborsClassifier(
                n_neighbors=k,
                weights=w
            )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, average='weighted')

            if f1 > best_score:
                best_score = f1
                best_model = model

    print(f"  Best Val F1: {best_score:.4f}")
    return best_model


# =========================
#  Models
# =========================

models_config = {
    "DecisionTree": {
        "trainer": train_decision_tree,
        "params": {
            "max_depth": [8, 10, 12],
            "min_samples_leaf": [1, 3, 5],
            "min_samples_split": [2, 5, 10]
        }
    },

    "RandomForest": {
        "trainer": train_random_forest,
        "params": {
            "n_estimators": [200, 300, 400],
            "max_depth": [15, 20, None],
            "min_samples_leaf": [1, 2, 3]
        }
    },

    "Logistic": {
        "trainer": train_logistic,
        "params": {
            "C": [0.01, 0.1, 1, 10, 100]
        }
    },

    "KNN": {
        "trainer": train_knn,
        "params": {
            "n_neighbors": list(range(1, 30, 2))
        }
    },

    "SVM": {
        "trainer": train_svm,
        "params": {
            "C": 30,
            "gamma": 0.01
        }
    }
}

