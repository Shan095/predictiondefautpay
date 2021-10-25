# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import plotly.express as px
from flask import Flask, request, render_template
import joblib
from joblib import load
import plotly.graph_objs as go
from plotly.subplots import make_subplots

app = Flask(__name__)
model = joblib.load('catboost_model_30f_2.joblib')
data=pd.read_csv('data.csv', sep=',')

@app.route('/', methods=['GET', 'POST'])  # une méthode de recevoir les données, à travers le serveur
def pred_model():
    return render_template("test.html")




if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000)
