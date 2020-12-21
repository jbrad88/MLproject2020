# flask for web app.
import flask as fl
# numpy for numerical work.
import numpy as np
# sklearn for machine learning
import sklearn.linear_model as lin
import pandas as pd
import seaborn as sns
import power
from flask import Flask, request, render_template, jsonify
#from power import prediction_model

# Load our data set.
df = pd.read_csv('powerproduction.csv')
#col=['speed','power']

# Create a new web app.
app = fl.Flask(__name__)

@app.route('/')
def index():
    return app.send_static_file('index.html')

#@app.route('/', methods =['POST'])
#def index_post():
#    ws_input = request.form['ws_input']
#    return power.user_input(ws_input)

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = power.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Sales should be $ {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = power.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)

# References
# [1] https://stackoverflow.com/questions/20646822/how-to-serve-static-files-in-flask
# https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4
