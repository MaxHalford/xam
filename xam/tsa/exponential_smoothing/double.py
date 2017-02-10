import pandas as pd

from .base import BaseExponentialSmoothingForecaster


class DoubleExponentialSmoothingForecaster(BaseExponentialSmoothingForecaster):

    def __init__(self, alpha, beta):
        # Parameters
        self.alpha = alpha
        self.beta = beta

        # Attributes
        self.smoothed_ = None
        self.trends_ = None
        self.fitted_ = None

    def fit(self, series):
        # Initialize the overall smoothing
        smoothed = pd.Series(index=series.index)
        smoothed[0] = series[0]

        # Initialize the trend
        trends = pd.Series(index=series.index)
        trends[0] = ((series[1] - series[0]) + (series[2] - series[1]) + (series[3] - series[2])) / 3

        # Initialize the fit
        fitted = pd.Series(index=series.index)
        fitted[0] = series[0]

        for i in range(1, len(smoothed)):

            # Calculate overall smoothing
            smoothed[i] = self._smooth(
                ratio=self.alpha,
                a=series[i],
                b=smoothed[i-1] + trends[i-1]
            )

            # Calculate trend smoothing
            trends[i] = self._smooth(
                ratio=self.beta,
                a=smoothed[i] - smoothed[i-1],
                b=trends[i-1]
            )

            fitted[i] = smoothed[i-1] + trends[i-1]

        self.smoothed_ = smoothed
        self.trends_ = trends
        self.fitted_ = fitted

        return self

    def predict(self, timestamps):
        forecasts = pd.Series(index=timestamps)

        for i in range(len(forecasts)):
            forecasts[i] = self.smoothed_[-1] + (i+1) * self.trends_[-1]

        return forecasts
