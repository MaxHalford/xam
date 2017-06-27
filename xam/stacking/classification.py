from sklearn import model_selection
from sklearn.base import ClassifierMixin

from .base import BaseStackingEstimator


class StackingClassifier(BaseStackingEstimator, ClassifierMixin):

    def __init__(self, models, meta_model, cv=model_selection.StratifiedKFold(n_splits=3),
                 use_base_features=True, use_proba=True):
        super().__init__(
            models=models,
            meta_model=meta_model,
            cv=cv,
            use_base_features=use_base_features,
            use_proba=use_proba,
        )
