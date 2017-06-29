import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

from xam.util import datetime_range


class DatetimeCV(BaseCrossValidator):

    """Cross-validation procedure that is aware of datetimes

    This goes a step further than sklearn's TimeSeriesSplit and takes into
    account the datetimes contained in the index of provided
    pandas.DataFrame. TimeSeriesSplit is more of a "running" cross-validation
    prodedure whereas DatetimeCV returns splits that correspond to datetimes.
    Moreover TimeSeriesSplit only produce one test index at a time. DatetimeCV
    will produce test indexes that match a given date.

    Attributes:
        timedelta (datetime.timedelta): the step to increase folds by.

    """

    def __init__(self, timedelta):
        super().__init__()
        self.timedelta = timedelta

    def split(self, X, y=None, groups=None):
        """

        Args:
            X (pd.DataFrame): a dataframe with a DatetimeIndex.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError('X is not a pandas.DataFrame')
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X's index is not a DatetimeIndex")

        min_dt = X.index.min()
        max_dt = X.index.max()

        indices = np.arange(len(X))

        for dt in datetime_range(min_dt+self.timedelta, max_dt, step=self.timedelta):
            train_idxs = indices[X.index < dt]
            test_idxs = indices[X.index == dt]

            if train_idxs.size == 0:
                raise ValueError('No data found before {}'.format(dt))
            if test_idxs.size == 0:
                raise ValueError('No data found for {}'.format(dt))

            yield train_idxs, test_idxs

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""
        if not isinstance(X, pd.DataFrame):
            raise ValueError('X is not a pandas.DataFrame')
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X's index is not a DatetimeIndex")

        first_dt = X.index.min()
        last_dt = X.index.max()

        return int((last_dt - first_dt) / self.timedelta)
