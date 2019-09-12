import collections

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import utils
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import clone
from sklearn.base import RegressorMixin
from sklearn.base import MetaEstimatorMixin


class BaseStackingEstimator(BaseEstimator, MetaEstimatorMixin):

    def __init__(self, models, meta_model, cv, metric, use_base_features, use_probas):
        self.models = models
        self.meta_model = meta_model
        self.cv = cv
        self.metric = metric
        self.use_base_features = use_base_features
        self.use_probas = use_probas

    def fit(self, X, y=None, verbose=False, **fit_params):

        # meta_features_ have as many rows as there are in X and as many
        # columns as there are models. However, if use_probas is True then
        # ((n_classes - 1) * n_models) columns have to be stored
        if self.use_probas:
            self.n_probas_ = len(np.unique(y)) - 1
            meta_features = np.empty((len(X), len(self.models) * (self.n_probas_)))
        else:
            meta_features = np.empty((len(X), len(self.models)))

        self.oof_scores_ = collections.defaultdict(list)

        if self.use_probas:
            lb = preprocessing.LabelBinarizer().fit(y)

        for i, (fit_idx, val_idx) in enumerate(self.cv.split(X, y)):
            for j, (name, model) in enumerate(self.models.items()):

                # Split the data according to the current folds
                if isinstance(X, pd.DataFrame):
                    X_fit, X_val = X.iloc[fit_idx], X.iloc[val_idx]
                else:
                    X_fit, X_val = X[fit_idx], X[val_idx]

                if isinstance(y, pd.Series):
                    y_fit, y_val = y.iloc[fit_idx], y.iloc[val_idx]
                else:
                    y_fit, y_val = y[fit_idx], y[val_idx]

                # Train the model on the training fold
                model.fit(X_fit, y_fit, **fit_params.get(name, {}))

                # If use_probas is True then the probabilities of each class for
                # each model have to be predicted and then stored into
                # meta_features
                if self.use_probas:
                    val_pred = model.predict_proba(X_val)
                    val_score = self.metric(y_val, lb.inverse_transform(val_pred))
                    a = self.n_probas_ * j
                    b = self.n_probas_ * (j + 1)
                    meta_features[val_idx, a:b] = val_pred[:, 0:self.n_probas_]
                else:
                    val_pred = model.predict(X_val)
                    meta_features[val_idx, j] = val_pred
                    val_score = self.metric(y_val, val_pred)

                # Store the model's score on the validation fold
                self.oof_scores_[name].append(val_score)

                if verbose:
                    print('OOF {} for fold {}: {:.5f}'.format(name, (i+1), val_score))

        if verbose:
            for name, scores in self.oof_scores_.items():
                print('OOF {} mean: {:.5f} (± {:.5f})'.format(name, np.mean(scores), np.std(scores)))

        # Combine the predictions with the original features
        if self.use_base_features:
            meta_features = np.hstack((meta_features, X))

        self.meta_model.fit(meta_features, y)

        # Each model has to be fit on all the data for further predictions
        for model in self.models.values():
            model.fit(X, y)

        return self

    def _predict(self, X, predict_proba):

        # If use_probas is True then the probabilities of each class for each
        # model have to be predicted and then stored into meta_features
        if self.use_probas:
            meta_features = np.empty((len(X), len(self.models) * (self.n_probas_)))
            for i, model in enumerate(self.models.values()):
                probabilities = model.predict_proba(X)
                for j, k in enumerate(range(self.n_probas_ * i, self.n_probas_ * (i + 1))):
                    meta_features[:, k] = probabilities[:, j]
        else:
            meta_features = np.transpose([model.predict(X) for model in self.models.values()])

        if self.use_base_features:
            meta_features = np.hstack((meta_features, X))

        if predict_proba:
            return self.meta_model.predict_proba(meta_features)
        return self.meta_model.predict(meta_features)

    def predict(self, X):
        return self._predict(X, predict_proba=False)


class StackingClassifier(BaseStackingEstimator, ClassifierMixin):

    def __init__(self, models, meta_model, cv=model_selection.StratifiedKFold(n_splits=3),
                 metric=metrics.roc_auc_score, use_base_features=False, use_probas=True):
        super().__init__(
            models=models,
            meta_model=meta_model,
            cv=cv,
            metric=metric,
            use_base_features=use_base_features,
            use_probas=use_probas,
        )

    def predict_proba(self, X):
        return self._predict(X, predict_proba=True)


class StackingRegressor(BaseStackingEstimator, RegressorMixin):

    def __init__(self, models, meta_model, cv=model_selection.KFold(n_splits=3),
                 metric=metrics.mean_squared_error, use_base_features=False):
        super().__init__(
            models=models,
            meta_model=meta_model,
            cv=cv,
            metric=metric,
            use_base_features=use_base_features,
            use_probas=False
        )


class BaggedStackingEstimator(BaseEstimator, MetaEstimatorMixin):

    def __init__(self, models, meta_model, cv, metric, use_base_features, use_probas, fit_handlers):
        self.models = models
        self.meta_model = meta_model
        self.cv = cv
        self.metric = metric
        self.use_base_features = use_base_features
        self.use_probas = use_probas
        self.fit_handlers = fit_handlers

    def fit(self, X, y=None, verbose=False):

        # meta_features_ is of shape (len(X), len(models)); if use_probas is
        # True then (n_classes - 1) columns have to be stored per model
        if self.use_probas:
            self.n_probas_ = len(np.unique(y)) - 1
            meta_features = np.empty((len(X), len(self.models) * (self.n_probas_)))
        else:
            meta_features = np.empty((len(X), len(self.models)))

        self.oof_scores_ = collections.defaultdict(list)
        self.instances_ = collections.defaultdict(list)

        if self.use_probas:
            lb = preprocessing.LabelBinarizer().fit(y)

        for i, (fit_idx, val_idx) in enumerate(self.cv.split(X, y)):
            for j, (name, model) in enumerate(self.models.items()):

                # Split the data according to the current folds
                if isinstance(X, pd.DataFrame):
                    X_fit, X_val = X.iloc[fit_idx], X.iloc[val_idx]
                else:
                    X_fit, X_val = X[fit_idx], X[val_idx]

                if isinstance(y, pd.Series):
                    y_fit, y_val = y.iloc[fit_idx], y.iloc[val_idx]
                else:
                    y_fit, y_val = y[fit_idx], y[val_idx]

                # Train the model on the training fold
                fit_handler = self.fit_handlers.get(name, lambda a, b, c, d: {})
                instance = clone(model)
                instance = instance.fit(X_fit, y_fit, **fit_handler(X_fit, y_fit, X_val, y_val))
                self.instances_[name].append(instance)

                # If use_probas is True then the probabilities of each class for
                # each model have to be predicted and then stored into
                # meta_features
                if self.use_probas:
                    val_pred = instance.predict_proba(X_val)
                    val_score = self.metric(y_val, lb.inverse_transform(val_pred))
                    a = self.n_probas_ * j
                    b = self.n_probas_ * (j + 1)
                    meta_features[val_idx, a:b] = val_pred[:, 0:self.n_probas_]
                else:
                    val_pred = instance.predict(X_val)
                    meta_features[val_idx, j] = val_pred
                    val_score = self.metric(y_val, val_pred)

                # Store the model's score on the validation fold
                self.oof_scores_[name].append(val_score)

                if verbose:
                    print('OOF {} for fold {}: {:.5f}'.format(name, (i+1), val_score))

        if verbose:
            for name, scores in self.oof_scores_.items():
                print('OOF {} mean: {:.5f} (± {:.5f})'.format(name, np.mean(scores), np.std(scores)))

        # Combine the predictions with the original features
        if self.use_base_features:
            meta_features = np.hstack((meta_features, X))

        # Train the meta-model
        self.meta_model  = self.meta_model.fit(meta_features, y)

        return self

    def _predict(self, X, predict_proba):

        utils.validation.check_is_fitted(self, ['oof_scores_', 'instances_'])

        # If use_probas is True then the probabilities of each class for each
        # model have to be predicted and then stored into meta_features
        if self.use_probas:
            meta_features = np.empty((len(X), len(self.models) * self.n_probas_))
            for i, name in enumerate(self.models):

                # Bag the predictions of each model instance
                instances = self.instances_[name]
                probabilities = np.mean([instance.predict_proba(X) for instance in instances], 0)

                # Add the predictions to the set of meta-features
                a = self.n_probas_ * i
                b = self.n_probas_ * (i + 1)
                meta_features[:, a:b] = probabilities[:, 0:self.n_probas_]

        else:
            # Bag the predictions of each model instance
            meta_features = np.transpose([
                np.mean([instance.predict(X) for instance in self.instances_[name]], 0)
                for name in self.models
            ])

        if self.use_base_features:
            meta_features = np.hstack((meta_features, X))

        if predict_proba:
            return self.meta_model.predict_proba(meta_features)
        return self.meta_model.predict(meta_features)

    def predict(self, X):
        return self._predict(X, predict_proba=False)


class BaggedStackingClassifier(BaggedStackingEstimator, ClassifierMixin):

    def __init__(self, models, meta_model, cv=model_selection.StratifiedKFold(n_splits=3),
                 metric=metrics.roc_auc_score, use_base_features=False, use_probas=True,
                 fit_handlers={}):
        super().__init__(
            models=models,
            meta_model=meta_model,
            cv=cv,
            metric=metric,
            use_base_features=use_base_features,
            use_probas=use_probas,
            fit_handlers=fit_handlers
        )

    def predict_proba(self, X):
        return super()._predict(X, predict_proba=True)


class BaggedStackingRegressor(BaggedStackingEstimator, RegressorMixin):

    def __init__(self, models, meta_model, cv=model_selection.KFold(n_splits=3),
                 metric=metrics.mean_squared_error, use_base_features=False,
                 fit_handlers={}):
        super().__init__(
            models=models,
            meta_model=meta_model,
            cv=cv,
            metric=metric,
            use_base_features=use_base_features,
            use_probas=False,
            fit_handlers=fit_handlers
        )
