import pickle
from flask import Flask, request, jsonify, render_template, app
import numpy as np 
import pandas as pd 

app = Flask(__name__)
# Loads the model
lrmodel = pickle.load(open('C:\\Users\\Admin\\Desktop\\Rohit\\MachineLearning\\fuel-consumption-prediction\\linear_regression_model.pkl', 'rb'))
scaler = pickle.load(open('C:\\Users\\Admin\\Desktop\\Rohit\\MachineLearning\\fuel-consumption-prediction\\scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict-fuel-consumed', methods = ['POST'])
def predict_fuel_consumed():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    output = lrmodel.predict(new_data)
    print(output)
    return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug = True)
