import numpy as np
from sklearn import model_selection
from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.utils.validation import check_X_y


class BaseStackingEstimator(BaseEstimator, MetaEstimatorMixin):

    def __init__(self, models, meta_model, n_folds, stratified, verbose):
        self.models = models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.stratified = stratified
        self.verbose = verbose

        self.meta_features = None

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # The meta features has as many rows as there are in X and as many columns as models
        self.meta_features = np.empty((len(X), len(self.models)))

        if self.stratified:
            folds = model_selection.StratifiedKFold(n_splits=self.n_folds).split(X, y)
        else:
            folds = model_selection.KFold(n_splits=self.n_folds).split(X)

        for i, (train_index, test_index) in enumerate(folds):

            if self.verbose:
                print('- Fold {} of {}'.format(i+1, self.n_folds))

            for j, estimator in enumerate(self.models):

                if self.verbose:
                    print('\t- Estimator {} of {}'.format(j+1, len(self.models)))

                # Train the model on the training set
                estimator.fit(X[train_index], y[train_index])
                # Store the predictions the model makes on the test set
                self.meta_features[test_index, j] = estimator.predict(X[test_index])

        self.meta_model.fit(self.meta_features, y)

        # Each model has to be fit on all the data for further predictions
        for model in self.models:
            model.fit(X, y)

        return self

    def predict(self, X):
        meta_features = np.transpose([model.predict(X) for model in self.models])
        return self.meta_model.predict(meta_features)
