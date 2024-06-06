import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFE



data = pd.read_csv("demmin.csv").to_numpy()
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

train, test = train_test_split(data, test_size=0.3)


x_train = train[:,1:]
x_test = test[:,1:]

y_train = train[:,0]
y_test = test[:,0]

model = RandomForestRegressor()
#fsm = SequentialFeatureSelector(model, n_features_to_select=10, direction='forward')
fsm = RFE(estimator=model, n_features_to_select=10)
fsm.fit(x_train, y_train)

x_train = fsm.transform(x_train)
x_test = fsm.transform(x_test)

model = RandomForestRegressor()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print(score)