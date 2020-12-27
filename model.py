#Import the librabries which we need to develop our model.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import requests
import json
import matplotlib.pyplot as plt 

# Load dataset. Note: We have cleansed the data set to take into account days on which the turbines were inactive.
# This has been determined to be days on which wind speeds were greater than 5mph, however, no power
# was generated. This was deeemed prudent in order to give a more accurate result.
dataset = pd.read_csv('powerproduction.csv')

# Take the two variables from our data set.
production = dataset[["speed", "power"]].dropna()

# Train the model
x = production["speed"].to_numpy()
y = production["power"].to_numpy()

x = x.reshape(-1, 1)

regressor = LinearRegression()
regressor.fit(x, y)

r = regressor.score(x, y)
p = [regressor.intercept_, regressor.coef_[0]]

# Predict. If we have the wind speed (x), how much power would be generated?
def f(x, p):
    return p[0] + x * p[1]

def predict(x, p):
    return f(x, p)

pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[15]]))