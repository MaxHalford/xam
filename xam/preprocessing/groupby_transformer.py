import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.base import TransformerMixin


class GroupbyTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, base_transformer, by):
        self.base_transformer = base_transformer
        self.by = by

    def _get_transform_columns(self, X):
        return [col for col in X.columns if col != self.by]

    def fit(self, X, y=None, **fit_params):

        if not isinstance(X, pd.DataFrame):
            raise ValueError('X is not a pandas.DataFrame')

        self.transformers_ = {}

        if y is None:
            y = np.zeros(shape=len(X))

        columns = self._get_transform_columns(X)

        for key in X[self.by].unique():

            # Copy the transformer
            transformer = clone(self.base_transformer)

            # Select the rows that will be fitted
            mask = (X[self.by] == key).tolist()
            rows = X.index[mask]

            # Fit the transformer
            transformer.fit(X.loc[rows, columns], y[mask], **fit_params)

            # Save the transformer
            self.transformers_[key] = transformer

        return self

    def transform(self, X):

        if not isinstance(X, pd.DataFrame):
            raise ValueError('X is not a pandas.DataFrame')

        columns = self._get_transform_columns(X)

        for key in X[self.by].unique():

            # Check if a transformer is associated with the key
            transformer = self.transformers_.get(key)
            if transformer is None:
                raise ValueError('No transformer is associated with key {}'.format(key))

            # Select the rows to transform
            mask = (X[self.by] == key).tolist()
            rows = X.index[mask]

            # Transform the rows
            X.iloc[rows, columns] = transformer.transform(X.loc[rows, columns])

        return X

    def fit_transform(self, X, y=None, **fit_params):

        if not isinstance(X, pd.DataFrame):
            raise ValueError('X is not a pandas.DataFrame')

        self.transformers_ = {}

        if y is None:
            y = np.zeros(shape=len(X))

        columns = self._get_transform_columns(X)

        for key in X[self.by].unique():

            # Copy the transformer
            transformer = clone(self.base_transformer)

            # Select the rows that will be fitted
            mask = (X[self.by] == key).tolist()
            rows = X.index[mask]

            # Fit and transform
            X.iloc[rows, columns] = transformer.fit_transform(
                X.iloc[rows, columns],
                y[mask],
                **fit_params
            )

            # Save the transformer
            self.transformers_[key] = transformer

        return X
