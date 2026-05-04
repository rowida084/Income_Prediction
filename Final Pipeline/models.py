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

    return best_model



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

