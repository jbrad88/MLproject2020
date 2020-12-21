import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
import joblib

dataset = pd.read_csv('powerproduction.csv')

#dataset['speed'].fillna(0, inplace=True)

#dataset['power'].fillna(dataset['power'].mean(), inplace=True)

#X = dataset.iloc[:, :3]


def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

col=['speed','power']
production = dataset[["speed", "power"]].dropna()
sns.regplot(x="speed", y="power", data=dataset)

import sklearn.linear_model as lin

#dataset.reshape(1, -1)

x = production["speed"].to_numpy()
y = production["power"].to_numpy()

model = lin.LinearRegression()
model.fit(x, y)

#r = model.score(x, y)
p = [model.intercept_, model.coef_[0]]

#def f(x, p):
#    return p[0] + x * p[1]

#def predict(x, p):
#    return f(x, p)
 
#result = predict(x, p)
#print(result)
#y = dataset.iloc[:, -1]

#from sklearn.linear_model import LinearRegression
#regressor = LinearRegression()

#regressor.fit(X, y)

#pickle.dump(predict, open('powerproduction.pkl','wb'))
#filename = 'modelz.pkl'
#outfile = open(filename,'wb')

#pickle.dump(dataset,outfile)
#outfile.close()

#pickle.dump(result, open('modelz.pkl','wb'))

#filename = 'modelz.pkl'
#pickle.dump(model, open(filename, 'wb'))

#loaded_model = pickle.load(open(filename, 'rb'))

#result = loaded_model.score(x, p)
#print(result)

#filename = 'finalised_model.sav'
#pickle.dump(model, open(filename, 'wb'))

#loaded_model = pickle.load(open(filename, 'rb'))
#result = 
#print(model.fit)

filename = 'finalized_model2.sav'
joblib.dump(model, filename)

loaded_model = joblib.load(filename)
result = loaded_model.score(x, y)
print(result)
