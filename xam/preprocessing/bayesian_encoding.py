import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class BayesianEncoder(BaseEstimator, TransformerMixin):

    """

    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf

    Args:
        columns (list of strs): Columns to encode.
        min_samples (int): Minimum samples.
        smoothing (float): Smoothing parameter.
        drop_columns (bool): Drop encoded columns or not.
    """

    def __init__(self, columns=None, min_samples=50, smoothing=5, drop_columns=True):
        self.columns = columns
        self.min_samples = min_samples
        self.smoothing = smoothing
        self.drop_columns = drop_columns

    def fit(self, X, y=None, **fit_params):

        if not isinstance(X, pd.DataFrame):
            raise ValueError('X has to be a pandas.DataFrame')

        if not isinstance(y, pd.Series):
            raise ValueError('y has to be a pandas.Series')

        dtypes = X.dtypes
        if self.columns is None:
            self.columns = [col for col in X.columns if dtypes[col] in ('object', 'category')]

        # Compute posterior probabilities for each feature
        features = pd.concat((X[self.columns], y.rename('y')), axis='columns')
        self.priors_ = {}
        self.posteriors_ = {}
        for col in self.columns:
            agg = features.groupby(col)['y'].agg(['count', 'mean'])
            counts = agg['count']
            means = agg['mean']
            self.priors_[col] = means.mean()
            p = 1 / (1 + np.exp(-(counts - self.min_samples) / self.smoothing))
            self.posteriors_[col] = p * means + (1 - p) * self.priors_[col]

        return self

    def transform(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            raise ValueError('X has to be a pandas.DataFrame')

        for col in self.columns:
            prior = self.priors_[col]
            posteriors = self.posteriors_[col]
            new_col = col if self.drop_columns else col + '_be'
            X[new_col] = X[col].apply(lambda x: posteriors.get(x, prior))

        return X
