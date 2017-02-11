import math

import numpy as np
import pandas as pd

from .base import BaseForecaster


class BaseExponentialSmoothingForecaster(BaseForecaster):

    def _smooth(self, ratio, left, right):
        return ratio * left + (1-ratio) * right


class SimpleExponentialSmoothingForecaster(BaseExponentialSmoothingForecaster):

    """Simple exponential smoothing."""

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
                left=series[i-1],
                right=smoothed[i-1]
            )

        self.smoothed_ = smoothed
        self.fitted_ = smoothed
        self.fitted_[0] = series[0]
        self.last_observation_ = series[-1]

        return self

    def predict(self, timestamps):
        forecasts = pd.Series(index=timestamps)

        forecasts[0] = self._smooth(
            ratio=self.alpha,
            left=self.last_observation_,
            right=self.smoothed_[-1]
        )

        for i in range(1, len(forecasts)):
            forecasts[i] = self._smooth(
                ratio=self.alpha,
                left=self.last_observation_,
                right=forecasts[i-1]
            )

        return forecasts


class DoubleExponentialSmoothingForecaster(BaseExponentialSmoothingForecaster):

    """Double exponential smoothing."""

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
        trends[0] = series[1] - series[0]

        # Initialize the fit
        fitted = pd.Series(index=series.index)
        fitted[0] = series[0]

        for i in range(1, len(smoothed)):

            # Calculate overall smoothing
            smoothed[i] = self._smooth(
                ratio=self.alpha,
                left=series[i],
                right=smoothed[i-1] + trends[i-1]
            )

            # Calculate trend smoothing
            trends[i] = self._smooth(
                ratio=self.beta,
                left=smoothed[i] - smoothed[i-1],
                right=trends[i-1]
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


class TripleExponentialSmoothingForecaster(BaseExponentialSmoothingForecaster):

    """Triple exponential smoothing.

    Also called the Holt-Winters method.

    Reference: http://www.itl.nist.gov/div898/handbook/pmc/section4/pmc435.htm
    """

    def __init__(self, alpha, beta, gamma, season_length, multiplicative=False):
        # Parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.season_length = season_length
        self.multiplicative = multiplicative

        # Attributes
        self.n_seasons_ = None
        self.smoothed_ = None
        self.trends_ = None
        self.seasonals_ = None
        self.fitted_ = None

    def fit(self, series):

        # TODO: Raise error if there are not 2 full seasons and values of alpha, beta, gamma etc.

        n = len(series) # Length of the series to fit
        k = self.season_length # Naming shortcut for the number of periods in each season
        p = self.n_seasons_ = math.ceil(n / k) # Number of seasons

        # Initialize the smoothed observations
        smoothed = pd.Series(index=series.index)
        smoothed[1] = series[0]

        # Initialize the trend
        trends = pd.Series(index=series.index)
        trends[1] = sum((series[i+k] - series[i]) / k for i in range(k)) / k

        # Initialize the seasonality
        seasonals = pd.Series(index=series.index)
        season_averages = [series[i*k:i*k+k].mean() for i in range(p)]

        for i in range(k):
            if self.multiplicative:
                seasonals[i] = sum(series[k*j+i] / season_averages[j] for j in range(p)) / p
            else:
                seasonals[i] = sum(series[k*j+i] - season_averages[j] for j in range(p)) / p

        # Initialize the fit
        fitted = pd.Series(index=series.index)
        fitted[:k] = series[:k]

        for i in range(2, n):

            # Calculate overall smoothing
            if i >= k:
                smoothed[i] = self._smooth(
                    ratio=self.alpha,
                    left=(
                        series[i] / seasonals[i-k]
                        if self.multiplicative
                        else series[i] - seasonals[i-k]
                    ),
                    right=smoothed[i-1] + trends[i-1]
                )
            else:
                smoothed[i] = self._smooth(
                    ratio=self.alpha,
                    left=series[i],
                    right=smoothed[i-1] + trends[i-1]
                )

            # Calculate trend smoothing
            trends[i] = self._smooth(
                ratio=self.beta,
                left=smoothed[i] - smoothed[i-1],
                right=trends[i-1]
            )

            # Calculate seasonal smoothing
            if i >= k:
                seasonals[i] = self._smooth(
                    ratio=self.gamma,
                    left=(
                        series[i] / smoothed[i]
                        if self.multiplicative
                        else series[i] - smoothed[i]
                    ),
                    right=seasonals[i-k]
                )

            # Calculate the one-sted ahead fitted value
            if i >= k:
                fitted[i] = (
                    smoothed[i] * seasonals[i-k]
                    if self.multiplicative
                    else smoothed[i] + seasonals[i-k]
                )

        self.smoothed_ = smoothed
        self.trends_ = trends
        self.seasonals_ = seasonals
        self.fitted_ = fitted

        return self

    def predict(self, timestamps):
        forecasts = pd.Series(index=timestamps)

        s = self.smoothed_[-1]
        t = self.trends_[-1]

        for i in range(len(forecasts)):
            season_look_back = -self.season_length + i % self.season_length
            if self.multiplicative:
                forecasts[i] = (s + (i+1) * t) * self.seasonals_[season_look_back]
            else:
                forecasts[i] = s + (i+1) * t + self.seasonals_[-self.season_length+i+1]

        return forecasts
