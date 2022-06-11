from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator

def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    train_score, validate_score = [], []
    chunks = np.mod(np.arange(len(X)), cv)
    for i in range(cv):
        X_train, y_train = X[chunks != i], y[chunks != i]
        X_validate, y_validate = X[chunks == i], y[chunks == i]
        model = estimator.fit(X_train, y_train)
        train_err = scoring(y_train, model.predict(X_train))
        validate_err = scoring(y_validate, model.predict(X_validate))
        train_score.append(train_err)
        validate_score.append(validate_err)
    return np.mean(train_score), np.mean(validate_score)

