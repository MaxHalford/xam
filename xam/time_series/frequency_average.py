import pandas as pd

from .base import BaseForecaster


class FrequencyAverageForecaster(BaseForecaster):

    """
    Args:
        timestamp_mapper (func): A function which converts a pandas.tslib.Timestamp to a value which
            will serve to group data.
    """

    def __init__(self, timestamp_mapper):
        # Parameters
        self.timestamp_mapper = timestamp_mapper

        # Attributes
        self.averages_ = None

    def fit(self, series):
        self.averages_ = series.groupby(self.timestamp_mapper).mean()
        return self

    def predict(self, timestamps):
        """Make forecasts from a list of datetimes.

        Args:
            datetimes (list(pandas.tslib.Timestamp))

        Returns:
            pandas.Series: A series with the provided datetimes as index and the corresponding
                forecasts as values.
        """
        forecasts = [
            self.averages_[self.timestamp_mapper(ts)]
            for ts in timestamps
        ]
        return pd.Series(data=forecasts, index=timestamps)
