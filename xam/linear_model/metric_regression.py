from functools import partial

import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn import linear_model


class ClassificationMetricRegression(BaseEstimator, ClassifierMixin):

    def __init__(self, metric):
        self.metric = metric

    def loss(self, coef, X, y):
        y_pred = sp.dot(X, coef)
        return self.metric(y, y_pred)

    def fit(self, X, y=None, **fit_params):
        lr = linear_model.LinearRegression()
        loss_partial = partial(self.loss, X=X, y=y)
        initial_coef = lr.fit(X, y).coef_
        self.coef_ = sp.optimize.fmin(loss_partial, initial_coef, disp=False)

    def predict(self, X):
        return sp.dot(X, self.coef_)

    def predict_proba(self, X):
        y_pred = sp.dot(X, self.coef_)
        y_pred /= np.max(y_pred)
        return np.vstack((1 - y_pred, y_pred)).T

    def score(self, X, y):
        return self.metric(y, sp.dot(X, self.coef_))
