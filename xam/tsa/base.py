class BaseForecaster():

    def fit(self, series):
        raise NotImplementedError

    def predict(self, timestamps):
        """Make forecasts from a list of timestamps.

        Args:
            timestamps (list(pandas.tslib.Timestamp))

        Returns:
            pandas.Series: A series with the provided datetimes as index and the corresponding
                forecasts as values.
        """
        raise NotImplementedError
