from sklearn.base import RegressorMixin

from .base import BaseStackingEstimator


class StackingRegressor(BaseStackingEstimator, RegressorMixin):

    def __init__(self, models, meta_model, n_folds=5, verbose=False):
        super().__init__(
            models=models,
            meta_model=meta_model,
            n_folds=n_folds,
            stratified=False,
            verbose=verbose
        )

    def check_params(self):
        if not isinstance(self.models, list):
            raise ValueError('models is not a list')
        for i, model in enumerate(self.models):
            if not isinstance(model, RegressorMixin):
                raise ValueError('Value %d of models is not a sklearn.base.RegressorMixin' % i)
        if not isinstance(self.meta_model, RegressorMixin):
            raise ValueError('meta_model is not a sklearn.base.RegressorMixin')
        return super().check_params()
