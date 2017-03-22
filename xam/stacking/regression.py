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
