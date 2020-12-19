# flask for web app.
import flask as fl
# numpy for numerical work.
import numpy as np
# sklearn for machine learning
import sklearn.linear_model as lin
import pandas as pd
import seaborn as sns
import power
from flask import Flask, request, render_template
#from power import prediction_model

# Load our data set.
df = pd.read_csv('powerproduction.csv')
#col=['speed','power']

# Create a new web app.
app = fl.Flask(__name__)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/', methods =['POST'])
def index_post():
    ws_input = request.form['ws_input']
    return power.prediction_model(df, ws_input)


# References
# [1] https://stackoverflow.com/questions/20646822/how-to-serve-static-files-in-flask
# https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4
