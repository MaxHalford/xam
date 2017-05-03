from sklearn.base import RegressorMixin

from .base import BaseStackingEstimator


class StackingRegressor(BaseStackingEstimator, RegressorMixin):

    def __init__(self, models, meta_model, n_folds=5, use_base_features=True):
        super().__init__(
            models=models,
            meta_model=meta_model,
            n_folds=n_folds,
            stratified=False,
            use_base_features=use_base_features
        )
