class BaseExponentialSmoothingForecaster():

    def _smooth(self, ratio, a, b):
        return ratio * a + (1-ratio) * b
