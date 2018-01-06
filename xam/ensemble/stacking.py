import numpy as np
from sklearn import metrics
from sklearn import model_selection
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.base import MetaEstimatorMixin


class BaseStackingEstimator(BaseEstimator, MetaEstimatorMixin):

    def __init__(self, models, meta_model, cv, metric, use_base_features, use_proba):
        self.models = models
        self.meta_model = meta_model
        self.cv = cv
        self.metric = metric
        self.use_base_features = use_base_features
        self.use_proba = use_proba

    def fit(self, X, y=None, verbose=False, **fit_params):

        # meta_features_ have as many rows as there are in X and as many
        # columns as there are models. However, if use_proba is True then
        # ((n_classes - 1) * n_models) columns have to be stored
        if self.use_proba:
            self.n_probas_ = len(np.unique(y)) - 1
            self.meta_features_ = np.empty((len(X), len(self.models) * (self.n_probas_)))
        else:
            self.meta_features_ = np.empty((len(X), len(self.models)))

        n_splits = self.cv.get_n_splits(X, y)
        self.scores_ = {
            name: [0] * n_splits
            for name in self.models.keys()
        }

        for i, (fit_idx, val_idx) in enumerate(self.cv.split(X, y)):
            for j, (name, model) in enumerate(self.models.items()):

                X_fit, y_fit = X.iloc[fit_idx], y.iloc[fit_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

                # Train the model on the training fold
                model.fit(X_fit, y_fit, **fit_params.get(name, {}))

                # If use_proba is True then the probabilities of each class for
                # each model have to be predicted and then stored into
                # meta_features
                if self.use_proba:
                    val_pred = model.predict_proba(X_val)
                    for k, l in enumerate(range(self.n_probas_ * j, self.n_probas_ * (j + 1))):
                        self.meta_features_[val_idx, l] = val_pred[:, k]
                else:
                    val_pred = model.predict(X_val)
                    self.meta_features_[val_idx, j] = val_pred

                # Score the model on the validation fold
                score = self.metric(y_val, val_pred)
                self.scores_[name][i] = score

                if verbose:
                    print('{} score on fold {}: {:.5f}'.format(name, (i+1), score))

        if verbose:
            for name, scores in self.scores_.items():
                print('{} mean score: {:.5f} (± {:.5f})'.format(name, np.mean(scores), np.std(scores)))

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
                 metric=metrics.roc_auc_score, use_base_features=False, use_proba=True):
        super().__init__(
            models=models,
            meta_model=meta_model,
            cv=cv,
            metric=metric,
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
                 metric=metrics.mean_squared_error, use_base_features=False):
        super().__init__(
            models=models,
            meta_model=meta_model,
            cv=cv,
            metric=metric,
            use_base_features=use_base_features,
            use_proba=False
        )
