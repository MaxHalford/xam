import numpy as np
from scipy import sparse
from sklearn import utils
from sklearn import linear_model


class NBSVMClassifier(linear_model.LogisticRegression):

    def predict(self, X):
        return super().predict(X.multiply(self.r_))

    def predict_proba(self, X):
        return super().predict_proba(X.multiply(self.r_))

    def fit(self, X, y, sample_weight=None):

        X, y = utils.check_X_y(X, y, accept_sparse='csr', order='C')

        def pr(x, y_i, y):
            p = x[y == y_i].sum(0)
            return (p+1) / ((y==y_i).sum()+1)

        self.r_ = sparse.csr_matrix(np.log(pr(X, 1, y) / pr(X, 0, y)))

        return super().fit(X.multiply(self.r_), y, sample_weight)
