# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import plotly.express as px
from flask import Flask, request, render_template
import joblib
from joblib import load
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import catboost
from catboost import CatBoostClassifier

app = Flask(__name__)
model = joblib.load('catboost_model_30f_2.joblib')
data=pd.read_csv('data.csv', sep=',')

@app.route('/', methods=['GET', 'POST'])  # une méthode de recevoir les données, à travers le serveur
def pred_model():
    return render_template("test.html")


@app.route('/model', methods=['GET', 'POST'])
def show_age():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('model.html')
    else:
        amount_credit = request.form['amt_credit']
        amt_annuity=request.form['amt_annuity']
        credit_length=request.form['credit_length']
        ext_source_2=request.form['ext_source_2']
        ext_source_3=request.form['ext_source_3']
        ext_source_1=request.form['ext_source_1']
        age=request.form['age']
        name_education_type=request.form['name_education_type']
        month_employed=request.form['months_employed']
        amt_credit=request.form['amt_credit']
        amt_goods_price=request.form['amt_goods_price']
        amt_annuity=request.form['amt_annuity']
        days_registration=request.form['days_registration']
        days_id_publish=request.form['days_id_publish']
        bureau_days_credit_enddate_max=request.form['bureau_days_credit_enddate_max']
        days_last_phone_change=request.form['days_last_phone_change']
        bureau_days_credit_mean=request.form['bureau_days_credit_mean']
        occupation_type=request.form['occupation_type']
        ratio_annuite_income=request.form['ratio_annuite_income']
        days_registration=request.form['days_registration']
        bureau_amt_credit_sum_sum=request.form['bureau_amt_credit_sum_sum']
        code_gender=request.form['code_gender']
        region_population_relative=request.form['region_population_relative']
        other_credits_count=request.form['other_credits_count']
        income_per_person=request.form['income_per_person']
        own_car_age=request.form['own_car_age']
        name_family_status=request.form['name_family_status']
        hour_appr_process_start=request.form['hour_appr_process_start']
        amt_req_credit_bureau_year=request.form['amt_req_credit_bureau_year']
        name_contract_type=request.form['name_contract_type']
        bureau_amt_credit_sum_overdue_count=request.form['bureau_amt_credit_sum_overdue_count']
        name_income_type=request.form['name_income_type']
        region_rating_client_w_city=request.form['region_rating_client_w_city']
        amt_goods_price=request.form['amt_goods_price']
        amt_income_total=request.form['amt_income_total']
        organization_type=request.form['organization_type']
        bureau_days_credit_enddate_max=request.form['bureau_days_credit_enddate_max']
        age=-float(age)*365
        days_employed=-float(month_employed)*30
        days_id_publish=-float(days_id_publish)
        days_registration=-float(days_registration)
        features_input=[ext_source_3, ext_source_2, credit_length, ext_source_1, age, days_employed, amt_goods_price, days_id_publish, name_education_type, amt_credit, amt_annuity, days_registration, days_last_phone_change, bureau_amt_credit_sum_sum, ratio_annuite_income, income_per_person, region_population_relative, own_car_age, occupation_type, other_credits_count, name_income_type, amt_income_total, organization_type, code_gender, name_family_status, hour_appr_process_start, amt_req_credit_bureau_year, name_contract_type, bureau_amt_credit_sum_overdue_count, region_rating_client_w_city]
        preds = model.predict_proba(features_input)
        #preds_as_str = str(preds[1])
        draw_subplots(data,
                      prediction=preds[1]*100,
                      months=month_employed,
                      ratio=ratio_annuite_income,
                      profession=occupation_type,
                      education=name_education_type,
                      family=name_family_status)
        return render_template('model.html')

y_train = pd.read_csv('y_train_catboost.csv', sep=',')
data['months_employed']=-data['days_employed']/12
data_profession=data.groupby(by='occupation_type')['target'].sum()
data_education=data.groupby(by='name_education_type')['target'].sum()
data_family=data.groupby(by='name_family_status')['target'].sum()

def draw_subplots(data,prediction,months,ratio,profession,education,family):
    fig = make_subplots(rows=3, cols=2,specs=[[{"type": "indicator"}, {"type": "xy"}],
                                              [{"type": "xy"}, {"type": "xy"}],
                                              [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=(None,
                        "Distribution default payers with months_employed",
                        "Distribution default payers with ratio_annuite_income",
                        "Sum of default payers with prefession",
                        "Sum of default payers with education_type",
                        "Sum of default payers with family_status"))
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = prediction,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Default risk score"},
        gauge = {'axis': {'range': [None, 100]},
                 'steps' : [
                     {'range': [0, 15], 'color': "greenyellow"},
                     {'range': [15, 100], 'color': "firebrick"}]}),
                  row=1, col=1)
    fig.add_trace(go.Box( x=data["target"], y=data["months_employed"]),
              row=1, col=2)
    fig.add_scatter(x=[0, 1], y=[months,months],mode="markers", marker=dict(size=5, color="red"),
                    showlegend=True,
                    row=1, col=2)
    fig.add_trace(go.Box( x=data["target"], y=data["ratio_annuite_income"]),
              row=2, col=1)
    fig.add_scatter(x=[0, 1], y=[ratio,ratio],mode="markers", marker=dict(size=5, color="red"),
                    showlegend=True,
                    row=2, col=1)
    fig.add_trace(go.Bar(x=data_profession.index,y=data_profession.values),
                  row=2, col=2)
    fig.add_scatter(x=[profession], y=[data_profession[profession].mean()],
                    mode="markers", marker=dict(size=5, color="red"),
                    showlegend=True,row=2,col=2)
    fig.add_trace(go.Bar(x=data_education.index,y=data_education.values),
                  row=3, col=1)
    fig.add_scatter(x=[education], y=[data_education[education].mean()],
                    mode="markers", marker=dict(size=5, color="red"),
                    showlegend=True,row=3,col=1)
    fig.add_trace(go.Bar(x=data_family.index,y=data_family.values),
                  row=3, col=2)
    fig.add_scatter(x=[family], y=[data_family[family].mean()],
                    mode="markers", marker=dict(size=5, color="red"),
                    showlegend=True,row=3,col=2)
    fig.update_layout(height=1000, width=1000, showlegend=True)
    fig.show()


if __name__ == "__main__":
    app.run(debug=True)
