import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.base import clone


class GroupbyModel(BaseEstimator, MetaEstimatorMixin):

    def __init__(self, base_model, by):
        self.base_model = base_model
        self.by = by

    def _get_fit_columns(self, X):
        return [col for col in X.columns if col != self.by]

    def fit(self, X, y=None, **fit_params):

        if not isinstance(X, pd.DataFrame):
            raise ValueError('X is not a pandas.DataFrame')

        self.models_ = {}

        columns = self._get_fit_columns(X)

        for key in X[self.by].unique():

            # Copy the model
            model = clone(self.base_model)

            # Select the rows that will be fitted
            mask = (X[self.by] == key).tolist()
            rows = X.index[mask]

            # Fit the model
            model.fit(X.loc[rows, columns], y[mask], **fit_params)

            # Save the model
            self.models_[key] = model

        return self

    def predict(self, X):

        if not isinstance(X, pd.DataFrame):
            raise ValueError('X is not a pandas.DataFrame')

        columns = self._get_fit_columns(X)
        y_pred = np.zeros(shape=len(X))

        for key in X[self.by].unique():

            # Check if a model is associated with the key
            model = self.models_.get(key)
            if model is None:
                raise ValueError('No model is associated with key {}'.format(key))

            # Select the rows to predict
            mask = (X[self.by] == key).tolist()
            rows = X.index[mask]

            # Predict the rows
            y_pred[mask] = model.predict(X.loc[rows, columns])

        return y_pred

    def predict_proba(self, X):

        if not isinstance(X, pd.DataFrame):
            raise ValueError('X is not a pandas.DataFrame')

        if not callable(getattr(self.base_model, 'predict_proba', None)):
            raise ValueError('base_model does not have any predict_proba method')

        columns = self._get_fit_columns(X)
        y_pred = np.zeros(shape=len(X))

        for key in X[self.by].unique():

            # Check if a model is associated with the key
            model = self.models_.get(key)
            if model is None:
                raise ValueError('No model is associated with key {}'.format(key))

            # Select the rows to predict
            mask = (X[self.by] == key).tolist()
            rows = X.index[mask]

            # Predict the rows
            y_pred[mask] = model.predict_proba(X.loc[rows, columns])

        return y_pred
