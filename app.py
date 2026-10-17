import pickle 
from flask import Flask, request, jsonify,url_for, render_template, redirect
import pandas as pd
import numpy as np

app=Flask(__name__, template_folder='template')
# Model Loading
regmodel=pickle.load(open('regmodel.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['post'])
def predict_api():
    data=request.json['data']
    # data is a list of records; take the first one
    if isinstance(data, list):
        data = data[0]
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])


@app.route('/predict',methods=['post'])
def predict():
    data=[float(x) for x in request.form.values()]
    new_data=scaler.transform(np.array(data).reshape(1,-1))
    output=regmodel.predict(new_data)
    return render_template('home.html', prediction_text='The predicted house price is ${:.2f}'.format(output[0]))

if __name__=="__main__":
    app.run(debug=True)