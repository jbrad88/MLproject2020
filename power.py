# Data set. Note: We have cleansed the data set to take into account days on which the turbines were inactive.
# This has been determined to be days on which wind speeds were greater than 5mph, however, no power
# was generated. This was deeemed prudent in order to give a more accurate result.

import pandas as pd
import numpy as np
import seaborn as sns
import sklearn.linear_model as lin

# Load dataset
df = pd.read_csv('powerproduction.csv')
col=['speed','power']

# Take the two variables from our data set.
production = df[["speed", "power"]].dropna()
# Scatter and fit line for just those two variables.
sns.regplot(x="speed", y="power", data=df)

def prediction_model(data,ws_input):
    ws_input = ws_input
    production = data

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

    return (predict(ws_input, p))

#def user_input(ws_input):
#    ws_input = ws_input
#    return prediction_model(df, ws_input)

print(prediction_model(15,0))

#windspeed = float(input("Enter value:"))
#print(predict(windspeed, p))

# this works when run on python. Takes user input and makes prediction.
