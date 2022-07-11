from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

LOW_UNIFORM = -1.2
HIGH_UNIFORM = 2
DEFAULT_MU = 0
K_DEGS = 10
N_TESTING_SAMPLES = 50
# TODO: Change
MIN_LAMBDA = 0.1
MAX_LAMBDA = 3

def model_func(x, eps):
    return (x+3)*(x+2)*(x+1)*(x-1)*(x-2) + eps 

def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    x = np.random.uniform(LOW_UNIFORM, HIGH_UNIFORM, size=n_samples)
    epsilon = np.random.normal(DEFAULT_MU, noise, size=n_samples)
    y = model_func(x, epsilon)
    X_train, y_train, X_test, y_test = split_train_test(pd.DataFrame(x), pd.Series(y), 2/3)
    X_train = np.array(X_train).reshape(1, -1)[0]
    X_test = np.array(X_test).reshape(1, -1)[0]
    y_train, y_test = np.array(y_train), np.array(y_test)
    fig = go.Figure(
        layout = go.Layout(
            title = f'Polynomial Function of {n_samples} Samples and Noise={noise}',
            xaxis_title = f'Samples ~ Unif[{LOW_UNIFORM}, {HIGH_UNIFORM}]',
            yaxis_title = 'Polynomial Function Values',
        )
    )
    fig.add_trace(
        go.Scatter(x=x, y=model_func(x, 0), mode='markers', 
        name='Model', showlegend=True)
    )
    fig.add_trace(
        go.Scatter(x=X_train, y=y_train, mode='markers', marker=dict(color='red'),
        name='Noisy Train', showlegend=True)
    )
    fig.add_trace(
        go.Scatter(x=X_test, y=y_test, mode='markers', marker=dict(color='green'), 
        name='Noisy Test', showlegend=True)
    )
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    
    train_errs, validate_errs = [], []
    for k in range(K_DEGS + 1):
        train_err, validate_err = cross_validate(
            PolynomialFitting(k), X_train, y_train, mean_square_error
            ) 
        train_errs.append(train_err)
        validate_errs.append(validate_err)
    
    fig = go.Figure(
        layout = go.Layout(
            title = 'Errors of Train and Validation Using 5-Fold Cross Validation',
            xaxis_title = 'Degree',
            yaxis_title = 'Mean Error',
        )
    )
    fig.add_trace(
        go.Scatter(x=np.arange(K_DEGS + 1), y=train_errs, 
        mode='lines+markers', name='Mean Train Error', showlegend=True)
    )
    fig.add_trace(
        go.Scatter(x=np.arange(K_DEGS + 1), y=validate_errs, 
        mode='lines+markers', name='Mean Validate Error', showlegend=True)
    )
    fig.show()

    best_validation_k = np.argmin(validate_errs) 
    
    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    model = PolynomialFitting(best_validation_k).fit(X_train, y_train)
    test_error = round(model.loss(X_test, y_test), 2)
    print(f'Best K for validation: {best_validation_k}')
    print(f'Test Error: {test_error}')

def append_model_error(model, value, train_lst, validate_lst, X, y,
scoring=mean_square_error, cv=5):
    train_err, validate_err = cross_validate(
        estimator = model(value),
        X = X, 
        y = y,
        scoring = scoring,
        cv = cv
    )
    train_lst.append(train_err)
    validate_lst.append(validate_err)
 

def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    X_train, y_train = X[:N_TESTING_SAMPLES, :], y[:N_TESTING_SAMPLES]
    X_test, y_test = X[N_TESTING_SAMPLES: ], y[N_TESTING_SAMPLES: ]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambda_range = np.linspace(MIN_LAMBDA, MAX_LAMBDA, n_evaluations)
    ridge_train_errs, ridge_validate_errs, lasso_train_errs, lasso_validate_errs = [], [], [], []
    for l in lambda_range:
        append_model_error(RidgeRegression, l, ridge_train_errs, ridge_validate_errs, X_train, y_train)
        append_model_error(Lasso, l, lasso_train_errs, lasso_validate_errs, X_train, y_train)
    
    fig = go.Figure(
        layout = go.Layout(
            title = 'Errors of Train and Validation in Ridge and Lasso as a Function of Parameter',
            xaxis_title = 'Parameter Value',
            yaxis_title = 'Mean Error',
        )
    )
    fig.add_trace(
        go.Scatter(x=lambda_range, y=ridge_train_errs, 
        mode='lines+markers', name='Ridge Train', showlegend=True)
    )
    fig.add_trace(
        go.Scatter(x=lambda_range, y=ridge_validate_errs, 
        mode='lines+markers', name='Ridge Validate', showlegend=True)
    )
    fig.add_trace(
        go.Scatter(x=lambda_range, y=lasso_train_errs, 
        mode='lines+markers', name='Lasso Train', showlegend=True)
    )
    fig.add_trace(
        go.Scatter(x=lambda_range, y=lasso_validate_errs, 
        mode='lines+markers', name='Lasso Validate', showlegend=True)
    )
    fig.show()
   
    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    ridge_best_lambda = MIN_LAMBDA + (np.argmin(ridge_validate_errs) / n_evaluations)
    lasso_best_lambda = MIN_LAMBDA + (np.argmin(lasso_validate_errs) / n_evaluations)
    ridge_model = RidgeRegression(ridge_best_lambda).fit(X_train, y_train)
    print(f'Ridge test error (lambda={ridge_best_lambda}): {mean_square_error(y_test, ridge_model.predict(X_test))}')
    # print(f'Ridge test error (lambda={ridge_best_lambda}): {ridge_model.loss(X_test, y_test)}')
    lasso_model = Lasso(lasso_best_lambda).fit(X_train, y_train)
    print(f'Lasso test error (lambda={lasso_best_lambda}): {mean_square_error(y_test, lasso_model.predict(X_test))}')
    lr_model = LinearRegression().fit(X_train, y_train)
    print(f'Linear Least Squares test error: {mean_square_error(y_test, lr_model.predict(X_test))}')

if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree(n_samples=100, noise=5)
    select_polynomial_degree(n_samples=100, noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter(n_samples=50, n_evaluations=500)