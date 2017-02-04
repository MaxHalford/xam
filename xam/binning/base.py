import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class BaseBinner(BaseEstimator, TransformerMixin):

    def __init__(self):
        # Attributes
        self.cut_points_ = None

    def transform(self, X, y=None):
        """Binarize X based on the fitted cut points."""
        X_discrete = np.array([
            np.digitize(X[:, i], self.cut_points_[i])
            for i in range(X.shape[1])
        ])
        return X_discrete
