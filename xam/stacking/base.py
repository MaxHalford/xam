import numpy as np
from sklearn import model_selection
from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.utils.validation import check_X_y


class BaseStackingEstimator(BaseEstimator, MetaEstimatorMixin):

    def __init__(self, models, meta_model, cv, use_base_features):
        self.models = models
        self.meta_model = meta_model
        self.cv = cv
        self.use_base_features = use_base_features

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # The meta features has as many rows as there are in X and as many columns as models
        self.meta_features_ = np.empty((len(X), len(self.models)))

        # Generate CV folds
        folds = self.cv.split(X, y)

        for train_index, test_index in folds:
            for j, model in enumerate(self.models):
                # Train the model on the training set
                model.fit(X[train_index], y[train_index])
                # Store the predictions the model makes on the test set
                self.meta_features_[test_index, j] = model.predict(X[test_index])

        if self.use_base_features:
            self.meta_features_ = np.hstack((self.meta_features_, X))

        self.meta_model.fit(self.meta_features_, y)

        # Each model has to be fit on all the data for further predictions
        for model in self.models:
            model.fit(X, y)

        return self

    def predict(self, X):
        meta_features = np.transpose([model.predict(X) for model in self.models])

        if self.use_base_features:
            meta_features = np.hstack((meta_features, X))

        return self.meta_model.predict(meta_features)
