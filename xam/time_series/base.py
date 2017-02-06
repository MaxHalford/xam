class BaseForecaster():

    def fit(self, series):
        raise NotImplementedError

    def forecast(self, n_steps):
        raise NotImplementedError
