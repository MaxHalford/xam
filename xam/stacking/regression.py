from sklearn import model_selection
from sklearn.base import RegressorMixin

from .base import BaseStackingEstimator


class StackingRegressor(BaseStackingEstimator, RegressorMixin):

    def __init__(self, models, meta_model, cv=model_selection.KFold(n_splits=3),
                 use_base_features=True):
        super().__init__(
            models=models,
            meta_model=meta_model,
            cv=cv,
            use_base_features=use_base_features,
            use_proba=False
        )
