class Model():

    def check_params(self):
        raise NotImplementedError

    @property
    def is_fitted(self):
        raise NotImplementedError
