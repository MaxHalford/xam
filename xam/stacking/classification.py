from sklearn.base import ClassifierMixin

from .base import BaseStackingEstimator


class StackingClassifier(BaseStackingEstimator, ClassifierMixin):

    def __init__(self, models, meta_model, n_folds=5, stratified=True, verbose=False):
        super().__init__(
            models=models,
            meta_model=meta_model,
            n_folds=n_folds,
            stratified=stratified,
            verbose=verbose
        )

    def check_params(self):
        if not isinstance(self.models, list):
            raise ValueError('models is not a list')
        for i, model in enumerate(self.models):
            if not isinstance(model, ClassifierMixin):
                raise ValueError('Value %d of models is not a sklearn.base.ClassifierMixin' % i)
        if not isinstance(self.meta_model, ClassifierMixin):
            raise ValueError('meta_model is not a sklearn.base.ClassifierMixin')
        return super().check_params()
