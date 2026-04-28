import numpy as np
from collections import Counter

def euclidean_distance(a,b):
    return np.sqrt(np.sum((a-b)**2))

class KNN:
  def __init__(self,k):
        self.k=k

  def fit(self,x,y):
    self.x_train=x
    self.y_train=y

  def predict(self,x):
    prediction = [self._predict(xi) for xi in x]
    return prediction

  def _predict(self,x):
  #  distances= [euclidean_distance(x,x_train) for x_train in self.x_train]
    distances = np.linalg.norm(self.x_train - x, axis=1)
    k_indices= np.argsort(distances)[:self.k]
    k_labels= [self.y_train[i] for i in k_indices]
    most_common= Counter(k_labels).most_common(1)
    return most_common[0][0]

#main
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# load data
train = pd.read_csv("processed_train_hott_data.csv")
test = pd.read_csv("processed_test_hott_data.csv")

# split features and target
  
x_train = train.drop("Income", axis=1)
y_train = train["Income"]

x_test = test.drop("Income", axis=1)
y_test = test["Income"]

x_train = x_train.values
y_train = y_train.values

x_test = x_test.values
y_test = y_test.values
#scaling 
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


model = KNN(k=29)
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print ("accuracy",accuracy_score(y_test,predictions))
