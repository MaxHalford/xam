import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.base import MetaEstimatorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted


class SplittingEstimator(BaseEstimator, MetaEstimatorMixin):

    def __init__(self, estimator, splitter):
        self.estimator = estimator
        self.splitter = splitter

    def _get_keys(self, X):

        # Map each row to a key according to the splitting function
        if isinstance(X, pd.DataFrame):
            keys = X.apply(self.splitter, axis='columns')
        else:
            keys = np.apply_along_axis(self.splitter, 1, X)

        return keys

    def fit(self, X, y=None, **fit_params):

        keys = self._get_keys(X)
        self.split_keys_ = np.unique(keys)

        # Create one copy of the estimator per splitting key
        self.estimators_ = {key: clone(self.estimator) for key in self.split_keys_}

        for key in self.split_keys_:
            mask = keys == key
            self.estimators_[key].fit(X[mask], y[mask], **fit_params)

        return self

    def predict(self, X):

        keys = self._get_keys(X)
        yy = np.zeros(shape=(X.shape[0],))
        yy[:] = np.nan

        for key in self.split_keys_:
            mask = keys == key
            yy[mask] = self.estimators_[key].predict(X[mask])

        return yy

    def predict_proba(self, X):

        keys = self._get_keys(X)
        yy = np.zeros(shape=(X.shape[0], 2))
        yy[:] = np.nan

        for key in self.split_keys_:
            mask = keys == key
            yy[mask] = self.estimators_[key].predict_proba(X[mask])

        return yy
