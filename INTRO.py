import pandas as pd
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output  # pip install dash (version 2.0.0 or higher)
import numpy as np
import pandas as pd
from joblib import load
from plotly.subplots import make_subplots




app = Dash(__name__)

# -- Import and clean data (importing csv into pandas)
# df = pd.read_csv("intro_bees.csv")
df = pd.read_csv("X_train_catboost.csv")
y_train = pd.read_csv("y_train_catboost.csv")
df=df.copy()
df['target']=y_train['target']
data_profession=df.groupby(by='occupation_type')['target'].sum()
data_education=df.groupby(by='name_education_type')['target'].sum()
data_family=df.groupby(by='name_family_status')['target'].sum()
data_income=df.groupby(by='name_income_type')['target'].sum()
data_organisation=df.groupby(by='organization_type')['target'].sum()

model = load('catboost_model_30f_2.joblib')

# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div([

    html.H1("Defaut payment analysis web application", style={'text-align': 'center'}),

    html.Label("Education type"),
    dcc.Dropdown(
                 id="slct_education",
                 options=[
                     {"label": "Secondary / secondary special", "value": 'Secondary / secondary special'},
                     {"label": "Higher education", "value": 'Higher education'},
                     {"label": "Incomplete Higher", "value":'Incomplete Higher'},
                     {"label": "Lower secondary", "value": 'Lower secondary'},
                     {"label": "Academic degree", "value": 'Academic degree'}],
                 multi=False,
                 value='Lower secondary',
                 placeholder='Select education type',
                 style={'width': "40%"}
                 ),
    html.Label("Income type"),
    dcc.Dropdown(id="slct_income_type",
                 options=[
                     {"label": "Working", "value": 'Working'},
                     {"label": "Student", "value": 'Student'},
                     {"label": "Businessman", "value":'Businessman'},
                     {"label": "Pensioner", "value": 'Pensioner'},
                     {"label": "Unemployed", "value": 'Unemployed'}],
                 multi=False,
                 value='Pensioner',
                 placeholder='Select income type',
                 style={'width': "40%"}
                 ),
    html.Label("Family status"),
    dcc.Dropdown(id="slct_family_status",
                 options=[
                     {"label": "Married", "value": 'Married'},
                     {"label": "Single", "value": 'Single'},
                     {"label": "Businessman", "value":'Businessman'},
                     {"label": "Civil marraige", "value": 'Civil marriage'},
                     {"label": "Widow", "value": 'Widow'},
                     {"label": "Unknown", "value": 'Unknown'}],
                 multi=False,
                 value='Unknown',
                 placeholder='Select family status',
                 style={'width': "40%"}
                 ),

    html.Label("Profession type"),
    dcc.Dropdown(id="slct_profession_type",
                 options=[
                     {"label": "Laborers", "value": 'Laborers'},
                     {"label": "Core staff", "value": 'Core staff'},
                     {"label": "Accountants", "value":'Accountants'},
                     {"label": "Managers", "value": 'Managers'},
                     {"label": "Drivers", "value": 'Drivers'},
                     {"label": "Sales staff", "value": 'Sales staff'},
                     {"label": "Cleaning staff", "value": 'Cleaning staff'},
                     {"label": "Sales staff", "value": 'Sales staff'},
                     {"label": "IT staff", "value": 'IT staff'}],
                 multi=False,
                 value='IT staff',
                 placeholder='Select profession type',
                 style={'width': "40%"}
                 ),

    html.Label("Organisation type"),
    dcc.Dropdown(id="slct_organization_type",
             options=[
                 {"label": "Insurance", "value": 'Insurance'},
                 {"label": "Business Entity Type 3", "value": 'Business Entity Type 3'},
                 {"label": "Transport: type 4", "value":'Transport: type 4'},
                 {"label": "School", "value": 'School'},
                 {"label": "Trade: type 7", "value": 'Trade: type 7'},
                 {"label": "Industry: type 9", "value": 'Industry: type 9'},
                 {"label": "Self-employed", "value": 'Self-employed'},
                 {"label": "Advertising", "value": 'Advertising'},
                 {"label": "Military", "value": 'Military'}],
             multi=False,
                 value='Military',
             placeholder='Select organisation type',
             style={'width': "40%"}
             ),
    html.Label("Gender"),
    dcc.Dropdown(id="slct_gender",
                 options=[
                     {"label": "Female", "value": 0},
                     {"label": "Male", "value": 1}],
                 multi=False,
                 value=1,
                 placeholder='Select gender',
                 style={'width': "40%"}
                 ),
    html.Label("Annual Income"),
    dcc.Slider(id='annual_income',
        marks={i: '{}'.format(i) for i in range(0, 100000,10000)},
        min=0,
        max=100000,
        value=10000
    ),
    html.Label("Credit amount"),
    dcc.Slider(id='credit_amount',
        marks={i: '{}'.format(i) for i in range(0, 1000000,100000)},
        min=0,
        max=1000000,
        value=100000
    ),
    dcc.Graph(id='score_graph', figure={}),
    dcc.Graph(id='education_graph', figure={})

])


# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
@app.callback(
    [Output(component_id='score_graph',component_property='figure'),
     Output(component_id='education_graph',component_property='figure')],
    [Input(component_id='slct_education', component_property='value'),
     Input(component_id='slct_income_type', component_property='value'),
     Input(component_id='slct_family_status', component_property='value'),
     Input(component_id='slct_profession_type', component_property='value'),
     Input(component_id='slct_organization_type', component_property='value'),
     Input(component_id='slct_gender', component_property='value'),
     Input(component_id='annual_income', component_property='value'),
     Input(component_id='credit_amount', component_property='value'),]
)
def update_output(education, income_type,family_status,profession,organisation, gender,
                  annual_income,credit_amount):
    features_input=[1, 1, 1, 1, 1, 1, annual_income, 1, education, credit_amount, 1, 1,
                    1, 1, 1, 1, 1, 1, profession, 1, income_type, 1, organisation,
                    gender, family_status, 1, 1, 1, 1, 1]
    preds = model.predict_proba(features_input)
    prediction = preds[1]
    trace1=go.Indicator(
        mode = "gauge+number",
        value = prediction*100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Default risk score"},
        gauge = {'axis': {'range': [None, 100]},
                 'steps' : [
                     {'range': [0, 15], 'color': "greenyellow"},
                     {'range': [15, 100], 'color': "firebrick"}]})
    figure1= go.Figure(data=trace1)

    trace2=go.Box( x=[education, income_type,family_status,profession,organisation],
                   y=[data_education[education],
                      data_income[income_type],
                      data_family[family_status],
                      data_profession[profession],
                      data_organisation[organisation]])
    figure2= go.Figure(data=trace2)
    return figure1,figure2


if __name__ == '__main__':
    app.run_server(debug=True)