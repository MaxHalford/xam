import collections

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.base import MetaEstimatorMixin


class BaseStackingEstimator(BaseEstimator, MetaEstimatorMixin):

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

        if self.use_probas:
            lb = preprocessing.LabelBinarizer().fit(y)

        for i, (fit_idx, val_idx) in enumerate(self.cv.split(X, y)):
            for j, (name, model) in enumerate(self.models.items()):

                if isinstance(X, pd.DataFrame):
                    X_fit = X.iloc[fit_idx]
                    X_val = X.iloc[val_idx]
                else:
                    X_fit = X[fit_idx]
                    X_val = X[val_idx]

                if isinstance(y, pd.Series):
                    y_fit = y.iloc[fit_idx]
                    y_val = y.iloc[val_idx]
                else:
                    y_fit = y[fit_idx]
                    y_val = y[val_idx]

                # Train the model on the training fold
                fit_handler = self.fit_handlers.get(name, lambda a, b, c, d: {})
                model.fit(X_fit, y_fit, **fit_handler(X_fit, y_fit, X_val, y_val))

                # If use_probas is True then the probabilities of each class for
                # each model have to be predicted and then stored into
                # meta_features
                if self.use_probas:
                    val_pred = model.predict_proba(X_val)
                    val_score = self.metric(y_val, lb.inverse_transform(val_pred))
                    for k, l in enumerate(range(self.n_probas_ * j, self.n_probas_ * (j + 1))):
                        meta_features[val_idx, l] = val_pred[:, k]
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

        # Train the meta-model
        self.meta_model  = self.meta_model.fit(meta_features, y)

        # Each model has to be fit on all the data for further predictions
        for model in self.models.values():
            model.fit(X, y)

        return self

    def _predict(self, X, proba=False):

        # If use_probas is True then the probabilities of each class for each
        # model have to be predicted and then stored into meta_features
        if self.use_probas:
            meta_features = np.empty((len(X), len(self.models) * self.n_probas_))
            for i, model in enumerate(self.models.values()):
                probabilities = model.predict_proba(X)
                for j, k in enumerate(range(self.n_probas_ * i, self.n_probas_ * (i + 1))):
                    meta_features[:, k] = probabilities[:, j]
        else:
            meta_features = np.transpose([model.predict(X) for model in self.models.values()])

        if self.use_base_features:
            meta_features = np.hstack((meta_features, X))

        return self.meta_model.predict(meta_features)

    def predict(self, X):
        return self._predict(X, proba=False)


class StackingClassifier(BaseStackingEstimator, ClassifierMixin):

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
        return super()._predict(X, proba=True)


class StackingRegressor(BaseStackingEstimator, RegressorMixin):

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
