import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

data = pd.read_csv("train_cleaned_v1.csv")

# =========================
# Cleaning
# =========================
data.replace(['?', ' ?','? '], np.nan, inplace=True)
data.rename(columns={"Income ": "Income"}, inplace=True)

for col in data.select_dtypes(include='object').columns:
    data[col] = data[col].str.strip()

data['Income'] = data['Income'].str.replace('.', '', regex=False)

# =========================
# Numeric handling
# =========================
numeric_cols = [
    "age", "education-num", "capital-gain",
    "capital-loss", "hours-per-week"
]

for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    data[col] = data[col].fillna(data[col].median())


for col in data.select_dtypes(include='object').columns:
    data[col] = data[col].fillna(data[col].mode()[0])

# =========================
# Feature Engineering
# =========================
data['native-country'] = data['native-country'].apply(
    lambda x: 'USA' if x == 'United-States' else 'Other'
)

data['net_capital'] = data['capital-gain'] - data['capital-loss']

data['has_gain'] = (data['capital-gain'] > 0).astype(int)

def work_status(x):
    if x < 35:
        return "part-time"
    elif x <= 45:
        return "full-time"
    else:
        return "over-time"

data['work_status'] = data['hours-per-week'].apply(work_status)

work_map = {
    "part-time": 0,
    "full-time": 1,
    "over-time": 2
}
data['work_status'] = data['work_status'].map(work_map)

# =========================
# Encoding
# =========================

# sex
data['sex'] = data['sex'].map({'Male': 1, 'Female': 0})

# target
data['Income'] = data['Income'].map({'>50K':1,'<=50K':0})

#  Frequency Encoding (occupation)
freq = data['occupation'].value_counts(normalize=True)
data['occupation'] = data['occupation'].map(freq)

# One-hot 
categorical_cols = [
    "workclass", "marital-status",
    "relationship", "native-country"
]

data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# =========================
# Outliers
# =========================
Q1 = data['age'].quantile(0.25)
Q3 = data['age'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

data['age'] = data['age'].clip(lower, upper)


data['capital-gain'] = np.log1p(data['capital-gain'])
data['capital-loss'] = np.log1p(data['capital-loss'])

# 6. Remove Duplicates

data = data.drop_duplicates()


# =========================
# Scaling
# =========================
scaler = StandardScaler()

numeric_cols += ["work_status","net_capital"]

data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# =========================
# Save
# =========================
medians = {}
modes = {}

for col in data.columns:
    if data[col].dtype == 'object':
        modes[col] = data[col].mode()[0]
    else:
        medians[col] = data[col].median()


training_columns = data.columns

preprocessing_objects = {
    "medians": medians,
    "modes": modes,
    "freq" : freq,
    "scaler": scaler,
    "columns": training_columns,
    "age_bounds": (lower, upper)
}

joblib.dump(preprocessing_objects, "preprocessing.pkl")

data.to_csv("processed_train_freq_data.csv", index=False)
