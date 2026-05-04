# =========================
#  Models
# =========================
models_config = {
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42,max_features='sqrt',class_weight='balanced',random_state=42,n_jobs=-1),
        "params": {
            "model__n_estimators": [200, 300, 400],
            "model__max_depth": [15, 20, None],
            "model__min_samples_leaf": [1, 2, 3]
        }
    },

    "Logistic": {
        "model": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "params": {
            "model__C": [0.01, 0.1, 1, 10, 100]
        }
    },

    "DecisionTree": {
        "model": DecisionTreeClassifier(random_state=42, criterion='entropy'),
        "params": {
            "model__max_depth": [8, 10, 12],
            "model__min_samples_leaf": [1, 3, 5],
            "model__min_samples_split" : [2, 5, 10]
        }
    },
    
    "KNN": {
        "model": KNeighborsClassifier(weights='distance'),
        "params": {
            "model__n_neighbors": list(range(1, 30, 2))
        }
    },
    
    "SVM": {
        "model": SVC(kernel='rbf',gamma=0.01, random_state=42 ,C=30,class_weight={0:1,1:2},probability=True)
    }
}    
