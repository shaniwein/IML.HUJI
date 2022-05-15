from __future__ import annotations
from enum import unique
from typing import Tuple, NoReturn
# from ...base import BaseEstimator
from IMLearn.base import BaseEstimator
from IMLearn.metrics.loss_functions import misclassification_error
import collections
import numpy as np
from itertools import product

Threshold = collections.namedtuple('Threshold', ['threshold', 'error', 'feat_idx'])

class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.sign_ = 1
        min_thr = None
        for i, col in enumerate(X.T):
            pos_thr, pos_thr_err = self._find_threshold(col, y, self.sign_)
            neg_thr, neg_thr_err = self._find_threshold(col, y, -self.sign_)
            thr = neg_thr if neg_thr_err < pos_thr_err else pos_thr
            thr_err = neg_thr_err if neg_thr_err < pos_thr_err else pos_thr_err
            self.sign_ = -self.sign_ if neg_thr_err < pos_thr_err else self.sign_ 
            if not min_thr or thr_err < min_thr.error:
                min_thr = Threshold(threshold=thr, error=thr_err, feat_idx=i)
        self.threshold_ = min_thr.threshold
        self.j_ = min_thr.feat_idx
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        condition = X[:, self.j_] < self.threshold_
        return np.where(condition, -self.sign_, self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        '''
        min_loss = None
        for val in unique(values):
            err =  misclassification_error(labels, values) 
            if not min_loss or err < min_loss.error:
                min_loss = Threshold(threshold=val, error=err, feat_idx=None)
        return min_loss.threshold, min_loss.error
        '''
        # Sort the values and labels by the values order
        values_sorted_indexes = np.argsort(values)
        values = values[values_sorted_indexes]
        labels = labels[values_sorted_indexes]
        mis_error = misclassification_error(labels, values)
        # thr_err = np.sum(np.abs(labels[np.sign(labels) == sign]))
        threshold = np.concatenate([[-np.inf], (values[1:] + values[:-1])/2, [np.inf]])
        losses = np.append(mis_error, mis_error-np.cumsum(sign*labels))
        min_loss = np.argmin(losses)
        return (threshold[min_loss], losses[min_loss])

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self.predict(X))