import numpy as np
from sklearn import model_selection
from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.utils.validation import check_X_y
from tqdm import tqdm

from ..base import Model
from ..check import is_a_bool
from ..check import is_a_positive_int


class BaseStackingEstimator(BaseEstimator, MetaEstimatorMixin, Model):

    def __init__(self, models, meta_model, n_folds, stratified, verbose):
        # Parameters
        self.models = models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.stratified = stratified
        self.verbose = verbose

        # Attributes
        self.meta_features_ = None

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # The meta features has as many rows as there are in X and as many columns as models
        self.meta_features_ = np.empty((len(X), len(self.models)))

        if self.stratified:
            folds = model_selection.StratifiedKFold(n_splits=self.n_folds).split(X, y)
        else:
            folds = model_selection.KFold(n_splits=self.n_folds).split(X)

        for train_index, test_index in tqdm(folds, desc='CV loop', disable=not self.verbose):
            for j, model in enumerate(tqdm(self.models, desc='Model loop', disable=not self.verbose)):
                # Train the model on the training set
                model.fit(X[train_index], y[train_index])
                # Store the predictions the model makes on the test set
                self.meta_features_[test_index, j] = model.predict(X[test_index])

        self.meta_model.fit(self.meta_features_, y)

        # Each model has to be fit on all the data for further predictions
        for model in self.models:
            model.fit(X, y)

        return self

    def predict(self, X):
        meta_features = np.transpose([model.predict(X) for model in self.models])
        return self.meta_model.predict(meta_features)

    def check_params(self):
        if not is_a_positive_int(self.n_folds):
            raise ValueError('n_folds is not a positive int')
        if not is_a_bool(self.stratified):
            raise ValueError('stratified is not a positive bool')
        if not is_a_bool(self.verbose):
            raise ValueError('verbose is not a positive bool')
        return

    @property
    def is_fitted(self):
        return self.meta_features_ is not None
