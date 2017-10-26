import numpy as np
from sklearn import model_selection
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.base import MetaEstimatorMixin
from sklearn.utils.validation import check_X_y


class BaseStackingEstimator(BaseEstimator, MetaEstimatorMixin):

    def __init__(self, models, meta_model, cv, use_base_features, use_proba):
        self.models = models
        self.meta_model = meta_model
        self.cv = cv
        self.use_base_features = use_base_features
        self.use_proba = use_proba

    def fit(self, X, y=None, **fit_params):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # meta_features_ have as many rows as there are in X and as many
        # columns as there are models. However, if use_proba is True then
        # ((n_classes - 1) * n_models) columns have to be stored
        if self.use_proba:
            self.n_probas_ = len(np.unique(y)) - 1
            self.meta_features_ = np.empty((len(X), len(self.models) * (self.n_probas_)))
        else:
            self.meta_features_ = np.empty((len(X), len(self.models)))

        # Generate CV folds
        folds = self.cv.split(X, y)

        for train_index, test_index in folds:
            for i, (name, model) in enumerate(self.models.items()):
                # Extract fit params for the model
                model_fit_params = fit_params.get('fit_params', {}).get(name, {})
                # Train the model on the training set
                model.fit(X[train_index], y[train_index], **model_fit_params)
                # If use_proba is True then the probabilities of each class for
                # each model have to be predicted and then stored into
                # meta_features
                if self.use_proba:
                    probabilities = model.predict_proba(X[test_index])
                    for j, k in enumerate(range(self.n_probas_ * i, self.n_probas_ * (i + 1))):
                        self.meta_features_[test_index, k] = probabilities[:, j]
                else:
                    self.meta_features_[test_index, i] = model.predict(X[test_index])

        # Combine the predictions with the original features
        if self.use_base_features:
            self.meta_features_ = np.hstack((self.meta_features_, X))

        self.meta_model.fit(self.meta_features_, y)

        # Each model has to be fit on all the data for further predictions
        for model in self.models.values():
            model.fit(X, y)

        return self

    def predict(self, X):

        # If use_proba is True then the probabilities of each class for each
        # model have to be predicted and then stored into meta_features
        if self.use_proba:
            meta_features = np.empty((len(X), len(self.models) * (self.n_probas_)))
            for i, model in enumerate(self.models.values()):
                probabilities = model.predict_proba(X)
                for j, k in enumerate(range(self.n_probas_ * i, self.n_probas_ * (i + 1))):
                    meta_features[:, k] = probabilities[:, j]
        else:
            meta_features = np.transpose([model.predict(X) for model in self.models.values()])

        if self.use_base_features:
            meta_features = np.hstack((meta_features, X))

        return self.meta_model.predict(meta_features)


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

    def predict_proba(self, X):

        # If use_proba is True then the probabilities of each class for each
        # model have to be predicted and then stored into meta_features
        if self.use_proba:
            meta_features = np.empty((len(X), len(self.models) * (self.n_probas_)))
            for i, model in enumerate(self.models.values()):
                probabilities = model.predict_proba(X)
                for j, k in enumerate(range(self.n_probas_ * i, self.n_probas_ * (i + 1))):
                    meta_features[:, k] = probabilities[:, j]
        else:
            meta_features = np.transpose([model.predict(X) for model in self.models.values()])

        if self.use_base_features:
            meta_features = np.hstack((meta_features, X))

        return self.meta_model.predict_proba(meta_features)


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
