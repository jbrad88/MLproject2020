import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns

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

#X['speed'] = X['speed'].apply(lambda x : convert_to_int(x))

import sklearn.linear_model as lin

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

#y = dataset.iloc[:, -1]

#from sklearn.linear_model import LinearRegression
#regressor = LinearRegression()

#regressor.fit(X, y)

#pickle.dump(predict, open('powerproduction.pkl','wb'))
filename = 'modelz.pkl'
outfile = open(filename,'wb')

pickle.dump(dataset,outfile)
outfile.close()


#model = pickle.load(open('powerproduction.pkl','rb'))
#print(model.predict([[4, 300, 500]]))
