import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#%matplotlib inline
# byzhr old output



# 1.read data

data = pd.read_csv("train_cleaned_v1.csv")



#  2. Handle Missing Values

data.replace(['?', ' ?','? '], np.nan, inplace=True)

numeric_cols = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]

for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')
# remove spaces
for col in data.select_dtypes(include='object').columns:
    data[col] = data[col].str.strip()

# fill missing
# fill categorical columns
for col in data.select_dtypes(include='object').columns:
    data[col] = data[col].fillna(data[col].mode()[0])

# fill numeric columns (important 🔥)
for col in numeric_cols:
    data[col] = data[col].fillna(data[col].median())


# 3. Feature Cleaning


# fix column name لو فيه space
data.rename(columns={"Income ": "Income"}, inplace=True)

# reduce(grouping) native-country
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


work_map = {
    "part-time": 0,
    "full-time": 1,
    "over-time": 2
}
data['work_status'] = data['work_status'].map(work_map)

data['has_gain'] = (data['capital-gain'] > 0).astype(int)

freq = data['occupation'].value_counts().to_dict()

# 4. Encoding

# binary
data['sex'] = data['sex'].map({'Male': 1, 'Female': 0})

# target
data['Income'] = data['Income'].map({'>50K':1,'<=50K':0})

# one-hot
categorical_cols = [
    "workclass",
    "marital-status",
    "occupation",
    "relationship",
    "native-country"
]

data = pd.get_dummies(data, columns=categorical_cols, drop_first=False)

# 5. Handle Outliers

# clip
Q1 = data['age'].quantile(0.25)
Q3 = data['age'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

data['age'] = data['age'].clip(lower, upper)

# log transform
data['capital-gain'] = np.log1p(data['capital-gain'])
data['capital-loss'] = np.log1p(data['capital-loss'])


data['has_gain'] = (data['capital-gain'] > 0).astype(int)


# 6. Remove Duplicates

data = data.drop_duplicates()


# 7. Feature Scaling ( Standardization )

scaler = StandardScaler()

numeric_cols = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]

data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
import joblib

medians = {}
modes = {}

for col in data.columns:
    if data[col].dtype == 'object':
        modes[col] = data[col].mode()[0]
    else:
        medians[col] = data[col].median()

#freq = data['occupation'].value_counts().to_dict()

training_columns = data.columns

preprocessing_objects = {
    "medians": medians,
    "modes": modes,
    "freq": freq,
    "scaler": scaler,
    "columns": training_columns
}

joblib.dump(preprocessing_objects, "preprocessing.pkl")
