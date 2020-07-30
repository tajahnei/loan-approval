from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = load_model('Final RF Model 29July2020')
cols = ['account_status',
 'duration',
 'credit_history',
 'purpose',
 'credit_amount',
 'savings_account',
 'unemployed',
 'installment_rate',
 'personal_status_sex',
 'debtors/guarantors',
 'present_residence_since',
 'property',
 'age',
 'other_installment_plans',
 'housing',
 'credits',
 'job',
 'liable_for',
 'telephone',
 'foreign_worker']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = predict_model(model, data=data_unseen, round = 0)
    prediction = int(prediction.Label[0])
    return render_template('home.html',pred='Expected Bill will be {}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
