from functools import partial

import scipy as sp
from sklearn import linear_model
from sklearn import metrics


class AUCRegressor():

    def _auc_loss(self, coef, X, y):
        fpr, tpr, _ = metrics.roc_curve(y, sp.dot(X, coef))
        return -metrics.auc(fpr, tpr)

    def fit(self, X, y, verbose=False):
        lr = linear_model.LinearRegression()
        auc_partial = partial(self._auc_loss, X=X, y=y)
        initial_coef = lr.fit(X, y).coef_
        self.coef_ = sp.optimize.fmin(auc_partial, initial_coef, disp=verbose)

    def predict(self, X):
        return sp.dot(X, self.coef_)

    def score(self, X, y):
        fpr, tpr, _ = metrics.roc_curve(y, sp.dot(X, self.coef_))
        return metrics.auc(fpr, tpr)
