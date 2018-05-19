import collections

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class SmoothTargetEncoder(BaseEstimator, TransformerMixin):

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

    def fit(self, X, y=None, **fit_params):

        if not isinstance(X, pd.DataFrame):
            raise ValueError('X has to be a pandas.DataFrame')

        if not isinstance(y, pd.Series):
            raise ValueError('y has to be a pandas.Series')

        # Default to using all the categorical columns
        columns = [col for col in X.columns if X.dtypes[col] in ('object', 'category')]\
            if self.columns is None\
            else self.columns

        # Compute prior and posterior probabilities for each feature
        X = pd.concat((X[columns], y.rename('y')), axis='columns')
        self.prior_ = y.mean()
        self.posteriors_ = {}
        for col in columns:
            agg = X.groupby(col)['y'].agg(['count', 'mean'])
            counts = agg['count']
            means = agg['mean']
            pw = self.prior_weight
            self.posteriors_[col] = collections.defaultdict(
                lambda: self.prior_,
                ((pw * self.prior_ + counts * means) / (pw + counts)).to_dict()
            )

        return self

    def transform(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            raise ValueError('X has to be a pandas.DataFrame')

        for col in self.columns:
            posteriors = self.posteriors_[col]
            X[col + self.suffix] = X[col].map(posteriors)

        return X
