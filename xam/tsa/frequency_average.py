import pandas as pd

from .base import BaseForecaster


class FrequencyAverageForecaster(BaseForecaster):

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
