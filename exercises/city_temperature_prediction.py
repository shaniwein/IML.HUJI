import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting, polynomial_fitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"

LABEL_COL = 'Temp'
TEMP_LIMIT = -70

def preprocess_data(df):
    df = df.drop_duplicates().dropna()
    # TODO: Drop temp under/above some value? df.drop(df[df[''] < limit].index, inplace=True)
    df.drop(df[df['Temp'] < TEMP_LIMIT].index, inplace=True)
    df['DayOfYear'] = df['Date'].dt.dayofyear
    return df

def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'])
    df = preprocess_data(df)
    df.insert(len(df.columns)-1, LABEL_COL, df.pop(LABEL_COL))
    return df

if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data('../datasets/City_Temperature.csv')
    
    # Question 2 - Exploring data for specific country
    israel_data = df[df['Country'] == 'Israel']
    # TODO: Are colors discrete?
    px.scatter(
        israel_data, x=israel_data['DayOfYear'], y=israel_data['Temp'], color=israel_data['Year'],
        title = 'Israel Average Daily Temperature by Day of Year',
        labels = {'DayOfYear': 'Day of Year', 'Temp': 'Temperature'}
    )#.show()
    
    grouped_data = israel_data[['Month', 'Temp']].groupby('Month').Temp.agg('std')
    px.bar(
        grouped_data, x=grouped_data.index, y=grouped_data, 
        title='Israel Standard Deviation of Daily Temperature by Month',
        labels={'y': 'Temperature Standard Deviation'}
    )#.show()
    
    # Question 3 - Exploring differences between countries
    grouped_data = df.groupby(['Country', 'Month']).Temp.agg(['mean', 'std']).reset_index()
    px.line(
        grouped_data, x=grouped_data['Month'], y=grouped_data['mean'], error_y=grouped_data['std'], color=grouped_data['Country'],
        title = 'Average Monthly Temperature By Country (std error bars)',
        labels = {'mean': 'Average Temperature'}
    )#.show()
    
    # Question 4 - Fitting model for different values of `k`
    X_train, y_train, X_test, y_test = split_train_test(israel_data.drop(LABEL_COL, axis=1), israel_data[LABEL_COL], 0.75)
    losses = {}
    for k in range(1, 11):
        poly_model = PolynomialFitting(k)
        # TODO: Is this the right column?
        poly_model.fit(X_train['DayOfYear'], y_train)
        losses[k] = round(poly_model._loss(X_test['DayOfYear'], y_test), 2)
    losses_df = pd.DataFrame.from_dict(losses, orient='index', columns=['loss'])
    px.bar(losses_df, x=losses_df.index, y=losses_df['loss']).show() 

    # Question 5 - Evaluating fitted model on different countries
    chosen_k = 5
    poly_model = PolynomialFitting(chosen_k)
    poly_model._fit(israel_data['DayOfYear'], israel_data[LABEL_COL])
    model_errors = {}
    for country in df['Country'].unique():
        if country == 'Israel':
            continue
        country_data = df[df['Country'] == country]
        model_errors[country] = poly_model.loss(country_data['DayOfYear'], country_data[LABEL_COL])
    px.Bar(x=model_errors.keys(), y=model_errors.values()).show()