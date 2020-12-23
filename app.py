import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 6)

    if output < 10:
        return render_template('index.html', prediction_text='The estimated wind power is {}...  not much power today!'.format(output))

    if output < 90:
        return render_template('index.html', prediction_text='The estimated wind power is {}'.format(output))
    
    else:
        return render_template('index.html', prediction_text='The estimated wind power is {}... batton down the hatches!'.format(output))


    
                           

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)

# REFERENCES
# [1] https://hackernoon.com/deploy-a-machine-learning-model-using-flask-da580f84e60c