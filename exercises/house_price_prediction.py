from enum import unique
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

import os
from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"

DATASET_PATH = '/home/shani/iml_course/IML.HUJI/datasets/house_prices.csv'
DATE_FORMAT = '%Y%m%dT%H%M%S'
NON_ZERO_COLS = ['sqft_living', 'sqft_lot', 'yr_built', 'sqft_living15', 'sqft_lot15']
ZIPCODE_PREFIX = 'zipcode_'

def get_timestamp(x):
    try:
        return pd.to_datetime(x, format=DATE_FORMAT).timestamp()
    except ValueError:
        return None

def preprocess_data(df):
    df = df.drop_duplicates().dropna()
    df['timestamp'] = df['date'].apply(get_timestamp)
    df = df.dropna()
    df['day_of_year'] = df['date'].apply(lambda x: pd.to_datetime(x, format=DATE_FORMAT).day_of_year)
    df['max_renovated'] = df[['yr_renovated', 'yr_built']].max(axis=1)
    df = df.drop(['id', 'date', 'lat', 'long', 'yr_renovated'], axis=1)
    for col_name in df.columns:
        df.drop(df[df[col_name] < (1 if col_name in NON_ZERO_COLS else 0)].index, inplace=True)
    # Get rid of outstanding
    df.drop(df[df['bedrooms'] > 30].index, inplace=True)
    df = pd.get_dummies(df, prefix=ZIPCODE_PREFIX, columns=['zipcode'], drop_first=True)
    return df

def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    df = preprocess_data(df)
    return df.drop(['price'], axis=1), df['price']

def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for col_name, col_data in X.iteritems():
        if col_name.startswith(ZIPCODE_PREFIX):
            continue
        p_corr = np.cov(col_data, y)[0][1] / (np.std(col_data) * np.std(y))
        fig = px.scatter(
            x=col_data, y=y,
            title=f'Pearson Correlation between {col_name} and price: {p_corr}',
            labels={'x': col_name, 'y': 'Price'}
        )
        fig.write_image(os.path.join(output_path, f'{col_name}_plot.png'))

if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data(DATASET_PATH)

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    X_train, y_train, X_test, y_test = split_train_test(X, y, 0.75)
    
    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    all_loss_mean = np.array([])
    all_loss_stdv = np.array([])
    p_values = np.arange(10, 101)
    for p in p_values:
        p_loss = np.array([])
        for i in range(10):
            # X_sample = X_train.sample(frac=p/100, replace=False)
            # y_sample = y_train[y_train.index.isin(X_sample.index)]
            X_sample, y_sample, _, _= split_train_test(X_train, y_train, p/100)
            lr_model = LinearRegression()
            lr_model.fit(X_sample, y_sample)
            p_loss = np.append(p_loss, lr_model.loss(X_test, y_test))
        all_loss_mean = np.append(all_loss_mean, p_loss.mean())
        all_loss_stdv = np.append(all_loss_stdv, p_loss.std())
    fig = go.Figure(
        data = go.Scatter(x=p_values, y=all_loss_mean, marker_color='rgb(131, 90, 241)', name='loss(mean)'),
        layout = go.Layout(
            title = 'Mean Loss by Precentage p of Training Set',
            xaxis = {'title': 'Precentage of Training Set'},
            yaxis = {'title': 'Mean Loss'},
        )
    )
    fig.add_traces([
        go.Scatter(x=p_values, y=all_loss_mean+(2*all_loss_stdv), mode='lines', line_color='rgb(11, 231, 219)', fill='tonexty', name='mean(loss)+2*std(loss)'),
        go.Scatter(x=p_values, y=all_loss_mean-(2*all_loss_stdv), mode='lines', line_color='rgba(11, 231, 219)', fill='tonexty', name='mean(loss)-2*std(loss)'),
    ])
    fig.show()