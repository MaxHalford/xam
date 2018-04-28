import threading

import numpy as np
from sklearn import ensemble
from sklearn import metrics
from sklearn import neighbors
from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.externals import joblib
from sklearn.utils import validation


def accumulate_prediction(predict, X, weights, out, lock):
    prediction = predict(X, check_input=True)
    with lock:
        if len(out) == 1:
            out[0] += prediction * weights
        else:
            for i in range(len(out)):
                out[i] += prediction[i] * weights[i]


def accumulate_score(predict, X, y, neigh_idxs, metric, out):
    for i, neigh_idx in enumerate(neigh_idxs):
        out[i] = metric(y[neigh_idx], predict(X[neigh_idx]))


class LocalizedEnsemble(BaseEstimator, MetaEstimatorMixin):

    def __init__(self, ensemble, metric, test_ratio, neighbors_ratio, algorithm, random_state,
                 verbose):

        self.ensemble = ensemble
        self.metric = metric
        self.test_ratio = test_ratio
        self.neighbors_ratio = neighbors_ratio
        self.algorithm = algorithm
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):

        # Split the training set in two
        X_fit, self.X_val_, y_fit, self.y_val_ = model_selection.train_test_split(
            X,
            y,
            test_size=self.test_ratio,
            random_state=self.random_state
        )

        # Fit the nearest neighbours
        n_neighbors = int(self.neighbors_ratio * len(self.X_val_))
        self.nn_ = neighbors.NearestNeighbors(n_neighbors=n_neighbors, algorithm=self.algorithm)
        self.nn_.fit(self.X_val_)

        # Fit the ensemble
        self.ensemble.fit(X_fit, y_fit)

        return self

    def predict(self, X):

        validation.check_is_fitted(self, 'nn_')

        n_estimators = len(self.ensemble.estimators_)

        # Assign chunk of trees to jobs
        n_jobs = 4

        # Avoid storing the output of every estimator by summing them here
        if self.ensemble.n_outputs_ > 1:
            y_hat = np.zeros((X.shape[0], self.ensemble.n_outputs_), dtype=np.float64)
        else:
            y_hat = np.zeros(X.shape[0], dtype=np.float64)

        # Compute the local scores in parallel
        _, neigh_idxs = self.nn_.kneighbors(X)
        scores = np.zeros(shape=(len(X), n_estimators), dtype=np.float64)
        # joblib.Parallel(n_jobs=n_jobs, verbose=self.verbose, backend='threading')(
        #     joblib.delayed(accumulate_score)(
        #         predict=e.predict,
        #         X=self.X_val_,
        #         y=self.y_val_,
        #         neigh_idxs=neigh_idxs,
        #         metric=self.metric,
        #         out=scores[:, i]
        #     )
        #     for i, e in enumerate(self.ensemble.estimators_)
        # )

        for i, e in enumerate(self.ensemble.estimators_):
            accumulate_score(
                predict=e.predict,
                X=self.X_val_,
                y=self.y_val_,
                neigh_idxs=neigh_idxs,
                metric=self.metric,
                out=scores[:, i]
            )

        weights = 1 / (0.1 + scores)

        # Make predictions in parallel
        lock = threading.Lock()
        # joblib.Parallel(n_jobs=n_jobs, verbose=self.verbose, backend='threading')(
        #     joblib.delayed(accumulate_prediction)(
        #         predict=estimator.predict,
        #         X=X,
        #         weights=weights[:, i],
        #         out=[y_hat],
        #         lock=lock
        #     )
        #     for i, estimator in enumerate(self.ensemble.estimators_)
        # )

        for i, estimator in enumerate(self.ensemble.estimators_):
            accumulate_prediction(
                predict=estimator.predict,
                X=X,
                weights=weights[:, i],
                out=[y_hat],
                lock=lock
            )

        # Average the predictions
        y_hat /= weights.sum(axis=1)

        return y_hat


class LocalizedEnsembleRegressor(LocalizedEnsemble):

    def __init__(self, ensemble=None, metric=metrics.mean_squared_error, test_ratio=0.3,
                 neighbors_ratio=0.1, algorithm='ball_tree', random_state=None, verbose=False):

        super().__init__(ensemble, metric, test_ratio, neighbors_ratio, algorithm, random_state,
                         verbose)


import numpy as np
from sklearn import datasets
from sklearn import ensemble
from sklearn import model_selection
from sklearn import preprocessing
import pandas as pd


n_estimators = 30
random_state = 42

models = {
    'Locally weighted random forest': LocalizedEnsembleRegressor(
        ensemble=ensemble.RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state
        ),
        random_state=random_state
    ),
    'Random forest': ensemble.RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state
    )
}

X = pd.read_csv('rest_X.csv').values
y = pd.read_csv('rest_y.csv')
y = y[y.columns[0]].values

X, _, y, _ = model_selection.train_test_split(
    X,
    y,
    test_size=0.9,
    random_state=random_state
)

#cv = model_selection.RepeatedKFold(n_repeats=10, n_splits=3, random_state=random_state)
cv = model_selection.KFold(n_splits=10, random_state=random_state, shuffle=True)

for model_name, model in models.items():
    scores = model_selection.cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    print('MSE: {:.3f} (± {:.3f}) [{}]'.format(-np.mean(scores), np.std(scores), model_name))
