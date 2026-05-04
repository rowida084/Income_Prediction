import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for col in X.select_dtypes(include=["object", "string"]).columns:
            X[col] = X[col].fillna("Unknown").astype("string").str.strip()

        X['capital-gain'] = pd.to_numeric(X['capital-gain'], errors='coerce').fillna(0)
        X['capital-loss'] = pd.to_numeric(X['capital-loss'], errors='coerce').fillna(0)

        X['native-country'] = X['native-country'].apply(
            lambda x: 'USA' if x == 'United-States' else 'Other'
        )

        X['net_capital'] = X['capital-gain'] - X['capital-loss']

        X['work_status'] = X['hours-per-week'].apply(
            lambda x: 0 if x < 35 else (1 if x <= 45 else 2)
        )

        X['has_gain'] = (X['capital-gain'] > 0).astype(int)

        X['capital-gain'] = np.log1p(X['capital-gain'].clip(lower=0))
        X['capital-loss'] = np.log1p(X['capital-loss'].clip(lower=0))

        return X
