import joblib
import pandas as pd
numeric_cols = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]
preprocessing_objects = joblib.load("preprocessing.pkl")

medians = preprocessing_objects["medians"]
modes = preprocessing_objects["modes"]
freq = preprocessing_objects["freq"]
scaler = preprocessing_objects["scaler"]
training_columns = preprocessing_objects["columns"]

data = pd.read_csv("test_data.csv")
#missing value
for col in modes:
    if col in data.columns:
        data[col] = data[col].fillna(modes[col])

for col in medians:
    if col in data.columns:
        data[col] = data[col].fillna(medians[col])

#Feature Engineering

data['native-country'] = data['native-country'].apply(
    lambda x: 'USA' if x == 'United-States' else 'Other'
)

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

data['has_gain'] = (data['capital-gain'].fillna(0) > 0).astype(int)
#encoding 
data['sex'] = data['sex'].map({'Male': 1, 'Female': 0})


data['occupation'] = data['occupation'].map(freq)
data['occupation'] = data['occupation'].fillna(0)
# one_Hot
data = pd.get_dummies(data)
data = data.reindex(columns=training_columns, fill_value=0)##edit

#scaling 
data[numeric_cols] = scaler.transform(data[numeric_cols])
data.to_csv("processed_test_data.csv", index=False)
