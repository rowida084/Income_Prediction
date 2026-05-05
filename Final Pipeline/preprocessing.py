import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineering(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

         # drop 
        X = X.drop(columns=['fnlwgt', 'education'], errors='ignore')

        # clean
        X = X.replace(['?', ' ?', '? '], np.nan)
        for col in X.select_dtypes(include='object').columns:
            X[col] = X[col].str.strip()

        # numeric
        num_cols = ["age", "education-num", "capital-gain",
                    "capital-loss", "hours-per-week"]
        for col in num_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        # حفظ medians و modes من الـ train بس
        self.medians_ = {col: X[col].median() for col in num_cols}
        self.modes_   = {
            col: X[col].mode()[0]
            for col in X.select_dtypes(include='object').columns
            if not X[col].mode().empty
        }


        
        Q1 = X['age'].quantile(0.25)
        Q3 = X['age'].quantile(0.75)

        IQR = Q3 - Q1
        self.lower_ = Q1 - 1.5 * IQR
        self.upper_ = Q3 + 1.5 * IQR


        return self

    def transform(self, X):
        X = X.copy()

        # drop
        X = X.drop(columns=['fnlwgt', 'education'], errors='ignore')

        # clean
        X.replace(['?', ' ?', '? '], np.nan, inplace=True)

        for col in X.select_dtypes(include='object').columns:
            X[col] = X[col].str.strip()

        # numeric تحويل بس (مش fill هنا)
        num_cols = ["age", "education-num", "capital-gain",
                    "capital-loss", "hours-per-week"]
        for col in num_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        for col in num_cols:
            X[col] = X[col].fillna(self.medians_[col])

        for col in X.select_dtypes(include='object').columns:
            if col in self.modes_:
                X[col] = X[col].fillna(self.modes_[col])

        # feature engineering
        X['native-country'] = X['native-country'].apply(
            lambda x: 'USA' if x == 'United-States' else 'Other'
        )

        X['net_capital'] = X['capital-gain'] - X['capital-loss']

        X['work_status'] = X['hours-per-week'].apply(
            lambda x: 0 if x < 35 else (1 if x <= 45 else 2)
        )

        X['has_gain'] = (X['capital-gain'] > 0).astype(int)

        # binary
        X['sex'] = X['sex'].map({'Male': 1, 'Female': 0})

        # clip age
        X["age"] = X["age"].clip(self.lower_, self.upper_)

        # log transform
        X['capital-gain'] = np.log1p(X['capital-gain'].clip(lower=0))
        X['capital-loss'] = np.log1p(X['capital-loss'].clip(lower=0))

        return X
