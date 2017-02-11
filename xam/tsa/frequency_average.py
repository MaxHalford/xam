import pandas as pd

from .base import BaseForecaster
from ..base import Model


class FrequencyAverageForecaster(BaseForecaster, Model):

    """
    Args:
        transform_timestamp (func): A function which converts a pandas.tslib.Timestamp to a value
            with which the data will be grouped by.
    """

    def __init__(self, transform_timestamp):
        # Parameters
        self.transform_timestamp = transform_timestamp

        # Attributes
        self.averages_ = None

    def fit(self, series):
        self.averages_ = series.groupby(self.transform_timestamp).mean()
        return self

    def predict(self, timestamps):
        forecasts = [
            self.averages_[self.transform_timestamp(ts)]
            for ts in timestamps
        ]
        return pd.Series(data=forecasts, index=timestamps)

    def check_params(self):
        ts = pd.tslib.Timestamp(dt.datetime.now())
        error = ValueError("transform_timestamp is not a function that transforms a "
                           "pandas.tslib.Timestam to a value")
        try:
            val = self.transform_timestamp(ts)
            if val is None:
                raise error
        except TypeError:
            raise error

    @property
    def is_fitted(self):
        return self.averages_ is not None
