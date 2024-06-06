import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


data = pd.read_csv("demmin.csv").to_numpy()
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

train, test = train_test_split(data, test_size=0.3)
#model = SVR(C=1e5, kernel='rbf', gamma=1.)
model = RandomForestRegressor()

x_train = train[:,1:]
x_test = test[:,1:]

y_train = train[:,0]
y_test = test[:,0]

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print(score)