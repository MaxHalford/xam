import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

from xam.util import datetime_range


class OrderedCV(BaseCrossValidator):

    """Cross-validation procedure with order of indexes taken into account.

    Say you have an interval [a, b] and you want to make n splits with d test
    indexes at each split -- for example 7 days. Then DatetimeCV will
    return the n following splits:

    - [a, b - d], [b - d, b]
    - [a, b - 2*d], [b - 2*d, b - d]
    - ...
    - [a, b - (n-1)*d], [b - (n-1)*d, b - (n-2)*d]
    - [a, b - n*d], [b - n*d, (n-1)*b]

    Attributes:
        n_splits (int): the number of desired splits.
        delta (int or datetime.timedelta): the step to increase folds by.

    """

    def __init__(self, n_splits, delta):
        super().__init__()
        self.n_splits = n_splits
        self.delta = delta

    def split(self, X, y=None, groups=None):
        """

        Args:
            X (pd.DataFrame): a pd.DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError('X is not a pandas.DataFrame')

        min_dt = X.index.min()
        max_dt = X.index.max()

        indices = np.arange(len(X))

        for i in range(self.n_splits):
            t0 = min_dt
            t1 = max_dt - self.delta * (i + 1)
            t2 = max_dt - self.delta * i

            train_idxs = indices[(X.index >= t0) & (X.index <= t1)]
            test_idxs = indices[(X.index > t1) & (X.index <= t2)]

            if train_idxs.size == 0:
                raise ValueError('No data found in [{}, {}]'.format(t0, t1))
            if test_idxs.size == 0:
                raise ValueError('No data found in ({}, {}]'.format(t1, t2))

            yield train_idxs, test_idxs

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""
        return self.n_splits
