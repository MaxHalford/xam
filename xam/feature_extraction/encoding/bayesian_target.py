import functools

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class BayesianTargetEncoder(BaseEstimator, TransformerMixin):

    """
    Reference: https://www.wikiwand.com/en/Bayes_estimator#/Practical_example_of_Bayes_estimators

    Args:
        columns (list of strs): Columns to encode.
        weighting (int or dict): Value(s) used to weight each prior.
        suffix (str): Suffix used for naming the newly created variables.

    """

    def __init__(self, columns=None, prior_weight=100, suffix='_mean'):
        self.columns = columns
        self.prior_weight = prior_weight
        self.suffix = suffix
        self.prior_ = None
        self.posteriors_ = None

    def fit(self, X, y=None, **fit_params):

        if not isinstance(X, pd.DataFrame):
            raise ValueError('X has to be a pandas.DataFrame')

        if not isinstance(y, pd.Series):
            raise ValueError('y has to be a pandas.Series')

        X = X.copy()

        # Default to using all the categorical columns
        columns = (
            X.select_dtypes(['object', 'category']).columns
            if self.columns is None else
            self.columns
        )

        names = []
        for cols in columns:
            if isinstance(cols, list):
                name = '_'.join(cols)
                names.append('_'.join(cols))
                X[name] = functools.reduce(
                    lambda a, b: a.astype(str) + '_' + b.astype(str),
                    [X[col] for col in cols]
                )
            else:
                names.append(cols)

        # Compute prior and posterior probabilities for each feature
        X = pd.concat((X[names], y.rename('y')), axis='columns')
        self.prior_ = y.mean()
        self.posteriors_ = {}

        for name in names:
            agg = X.groupby(name)['y'].agg(['count', 'mean'])
            counts = agg['count']
            means = agg['mean']
            pw = self.prior_weight
            self.posteriors_[name] = ((pw * self.prior_ + counts * means) / (pw + counts)).to_dict()

        return self

    def transform(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            raise ValueError('X has to be a pandas.DataFrame')

        for cols in self.columns:

            if isinstance(cols, list):
                name = '_'.join(cols)
                x = functools.reduce(
                    lambda a, b: a.astype(str) + '_' + b.astype(str),
                    [X[col] for col in cols]
                )
            else:
                name = cols
                x = X[name]

            posteriors = self.posteriors_[name]
            X[name + self.suffix] = x.map(posteriors).fillna(self.prior_).astype(float)

        return X
