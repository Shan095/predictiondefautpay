import pandas as pd
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State  # pip install dash (version 2.0.0 or higher)
from dash.exceptions import PreventUpdate
import numpy as np
import pandas as pd
from joblib import load
from plotly.subplots import make_subplots




app = Dash(__name__)

# -- Import and clean data (importing csv into pandas)
# df = pd.read_csv("intro_bees.csv")
df = pd.read_csv("X_train_catboost.csv")
y_train = pd.read_csv("y_train_catboost.csv")
train=pd.read_csv("application_train.csv")
df=df.copy()
df['target']=y_train['target']
data_profession=df.groupby(by='occupation_type')['target'].sum()
data_education=df.groupby(by='name_education_type')['target'].sum()
data_family=df.groupby(by='name_family_status')['target'].sum()
data_income=df.groupby(by='name_income_type')['target'].sum()
data_organisation=df.groupby(by='organization_type')['target'].sum()
target_list=train['SK_ID_CURR'].unique()
list_options=[]
for i in target_list:
    options={f"label": f"{i}", "value":  f"{i}"}
    list_options.append(options)
model = load('catboost_model_30f_2.joblib')

# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div([

    html.H1("Defaut payment analysis web application", style={'text-align': 'center'}),
    html.H2("Part I: Is the client an old client with us?", style={'text-align': 'left'}),
    html.Label("Please enter the client id if he/she has one"),
    dcc.Dropdown(
        id="input_box",
        options=list_options,
        multi=False,
        value='id',
        placeholder='Select education type',
        style={'width': "40%"},
    ),
    html.Div(id='output_container', children=[]),
    html.Br(),
    html.H2("Part II: If he/she is a new client, please enter the client's information here", style={'text-align': 'left'}),
    html.Label("Education type"),
    dcc.Dropdown(
                 id="slct_education",
                 options=[
                     {"label": "Secondary / secondary special", "value": 'Secondary / secondary special'},
                     {"label": "Higher education", "value": 'Higher education'},
                     {"label": "Incomplete higher", "value":'Incomplete higher'},
                     {"label": "Lower secondary", "value": 'Lower secondary'},
                     {"label": "Academic degree", "value": 'Academic degree'}],
                 multi=False,
                 value='Lower secondary',
                 placeholder='Select education type',
                 style={'width': "40%"},
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
                     {"label": "Single / not married", "value": 'Single / not married'},
                     {"label": "Civil marriage", "value": 'Civil marriage'},
                     {"label": "Widow", "value": 'Widow'},
                     {"label": "Unknown", "value": 'Unknown'},
                     {"label": "Separated", "value": 'Separated'}],
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
    [Output(component_id='output_container',component_property='children'),
     Output(component_id='score_graph',component_property='figure'),
     Output(component_id='education_graph',component_property='figure')],
    [Input(component_id='input_box', component_property='value'),
     Input(component_id='slct_education', component_property='value'),
     Input(component_id='slct_income_type', component_property='value'),
     Input(component_id='slct_family_status', component_property='value'),
     Input(component_id='slct_profession_type', component_property='value'),
     Input(component_id='slct_organization_type', component_property='value'),
     Input(component_id='slct_gender', component_property='value'),
     Input(component_id='annual_income', component_property='value'),
     Input(component_id='credit_amount', component_property='value')]
)
def update_output(client_id,education, income_type,family_status,profession,organisation, gender,
                  annual_income,credit_amount):
    dff=train[train['SK_ID_CURR']==client_id]
    record_result=dff['TARGET'].values
    if record_result is None:
        raise dash.exceptions.PreventUpdate
    elif record_result==1:
        result_str="This client has defaut payment history! He/she is a risky client!"


    features_input=[1, 1, 1, 1, 1, 1, annual_income, 1, education, credit_amount, 1, 1,
                    1, 1, 1, 1, 1, 1, profession, 1, income_type, 1, organisation,
                    gender, family_status, 1, 1, 1, 1, 1]
    preds = model.predict_proba(features_input)
    prediction = preds[1]
    #first figure
    trace2=go.Indicator(
        mode = "gauge+number",
        value = prediction*100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Default risk score"},
        gauge = {'axis': {'range': [None, 100]},
                 'steps' : [
                     {'range': [0, 15], 'color': "greenyellow"},
                     {'range': [15, 100], 'color': "firebrick"}]})
    figure2= go.Figure(data=trace2)
    #second figure
    x=[education, income_type,family_status,profession,organisation]
    y=[data_education[education],
       data_income[income_type],
       data_family[family_status],
       data_profession[profession],
       data_organisation[organisation]]
    y_per=[]
    for i in y:
        i=i/df.shape[0]*100
        y_per.append(i)
    figure3=make_subplots(rows=2, cols=1,  shared_xaxes=True,
                    shared_yaxes=False, vertical_spacing=0.1,
                    subplot_titles=("Sum of defaut payers by input category",
                                    "Percentage of defaut payers by input category"))
    figure3.append_trace(go.Bar(x=x,y=y,marker=dict(
        color='rgba(50, 171, 96, 0.6)',
        line=dict(
            color='rgba(50, 171, 96, 1.0)',
            width=1) )), 1, 1)
    figure3.append_trace(go.Scatter(x=x, y=y_per,mode='lines+markers',line_color='rgb(128, 0, 128)'), 2, 1)
    #figure3.update_layout(height=500, width=800)
    return result_str,figure2,figure3


if __name__ == '__main__':
    app.run_server(debug=True)