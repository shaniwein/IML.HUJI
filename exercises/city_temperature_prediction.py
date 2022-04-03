import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"

LABEL_COL = 'Temp'

def preprocess_data(df):
    df = df.drop_duplicates().dropna()
    # TODO: Drop temp under/above some value? df.drop(df[df[''] < limit].index, inplace=True)
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
    px.scatter(israel_data, x=israel_data['DayOfYear'], y=israel_data['Temp'], color=israel_data['Year']).show()
    
    # TODO: Group by month
    # Question 3 - Exploring differences between countries
    raise NotImplementedError()

    # Question 4 - Fitting model for different values of `k`
    raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    raise NotImplementedError()