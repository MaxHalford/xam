import pandas as pd

from .base import BaseExponentialSmoothingForecaster


class SimpleExponentialSmoothingForecaster(BaseExponentialSmoothingForecaster):

    def __init__(self, alpha):
        # Parameters
        self.alpha = alpha

        # Attributes
        self.last_observation_ = None
        self.smoothed_ = None
        self.fitted_ = None

    def fit(self, series):
        # Initialize the smoothing
        smoothed = pd.Series(index=series.index)
        smoothed[1] = series[0]

        for i in range(2, len(smoothed)):
            smoothed[i] = self._smooth(
                ratio=self.alpha,
                a=series[i-1],
                b=smoothed[i-1]
            )

        self.smoothed_ = smoothed[1:]
        self.fitted_ = smoothed
        self.fitted_[0] = series[0]
        self.last_observation_ = series[-1]

        return self

    def predict(self, timestamps):
        forecasts = pd.Series(index=timestamps)

        forecasts[0] = self._smooth(
            ratio=self.alpha,
            a=self.last_observation_,
            b=self.smoothed_[-1]
        )

        for i in range(1, len(forecasts)):
            forecasts[i] = self._smooth(
                ratio=self.alpha,
                a=self.last_observation_,
                b=forecasts[i-1]
            )

        return forecasts
