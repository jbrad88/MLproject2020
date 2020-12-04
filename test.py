# flask for web app.
import flask as fl
# numpy for numerical work.
import numpy as np
# sklearn for machine learning
import sklearn.linear_model as lin
import pandas as pd
import seaborn as sns

df = pd.read_csv('powerproduction.csv')
col=['speed','power']

# Take the two variables from our data set.
production = df[["speed", "power"]].dropna()
# Scatter and fit line for just those two variables.
sns.regplot(x="speed", y="power", data=df)

x = production["speed"].to_numpy()
y = production["power"].to_numpy()

x = x.reshape(-1, 1)

model = lin.LinearRegression()
model.fit(x, y)

r = model.score(x, y)
p = [model.intercept_, model.coef_[0]]

def f(x, p):
    return p[0] + x * p[1]
def predict(x, p):
    return f(x, p)

print(predict(15,p))