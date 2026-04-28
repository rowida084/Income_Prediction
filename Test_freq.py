import pandas as pd
import numpy as np
import joblib

preprocessing_objects = joblib.load("preprocessing.pkl")

medians = preprocessing_objects["medians"]
modes = preprocessing_objects["modes"]
freq = preprocessing_objects["freq"]
scaler = preprocessing_objects["scaler"]
training_columns = preprocessing_objects["columns"]
lower, upper = preprocessing_objects["age_bounds"]

data = pd.read_csv("test_data.csv")

# Cleaning
data.replace(['?', ' ?','? '], np.nan, inplace=True)

data.rename(columns={"Income ": "Income"}, inplace=True)

for col in data.select_dtypes(include='object').columns:
    data[col] = data[col].str.strip()

data['Income'] = data['Income'].str.replace('.', '', regex=False)

# Fill missing
for col in modes:
    if col in data.columns:
        data[col] = data[col].fillna(modes[col])

for col in medians:
    if col in data.columns:
        data[col] = data[col].fillna(medians[col])


numeric_cols = [
    "age", "education-num", "capital-gain",
    "capital-loss", "hours-per-week"
]

for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    data[col] = data[col].fillna(data[col].median())

# Features
data['native-country'] = data['native-country'].apply(
    lambda x: 'USA' if x == 'United-States' else 'Other'
)

data['net_capital'] = data['capital-gain'] - data['capital-loss']

def work_status(x):
    if x < 35:
        return "part-time"
    elif x <= 45:
        return "full-time"
    else:
        return "over-time"

data['work_status'] = data['hours-per-week'].apply(work_status)

work_map = {"part-time": 0, "full-time": 1, "over-time": 2}
data['work_status'] = data['work_status'].map(work_map)

data['has_gain'] = (data['capital-gain'] > 0).astype(int)

# Encoding
data['sex'] = data['sex'].map({'Male': 1, 'Female': 0})
data['Income'] = data['Income'].map({'>50K':1,'<=50K':0})

# freq encoding
data['occupation'] = data['occupation'].map(freq).fillna(0) # more accuracy than freq.mean() idk how

# one-hot
data = pd.get_dummies(data)

data = data.reindex(columns=training_columns, fill_value=0)

#outliers

data['age'] = data['age'].clip(lower, upper)

data['capital-gain'] = np.log1p(data['capital-gain'])
data['capital-loss'] = np.log1p(data['capital-loss'])

# scaling

numeric_cols += ["work_status","net_capital"]
data[numeric_cols] = scaler.transform(data[numeric_cols])

data.to_csv("processed_test_freq_data.csv", index=False)
