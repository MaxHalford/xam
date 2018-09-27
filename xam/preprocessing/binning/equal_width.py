"""
Equal width binning
"""

import numpy as np
from sklearn.utils import check_array

from .base import BaseUnsupervisedBinner


class EqualWidthBinner(BaseUnsupervisedBinner):

    def __init__(self, n_bins=5):

        super().__init__()

        # Properties
        self.n_bins = n_bins

    def fit(self, X, y=None, **fit_params):
        """Choose equally spaces cut points."""

        # scikit-learn checks
        X = check_array(X)

        self.cut_points_ = [0] * X.shape[1]

        for i, x in enumerate(X.T):
            x_min = np.min(x)
            x_max = np.max(x)

            if x_min == x_max:
                self.cut_points_[i] = np.array([x_min])
            else:
                step = (x_max - x_min) / self.n_bins
                self.cut_points_[i] = np.arange(start=x_min+step, stop=x_max, step=step).tolist()

        return self

    @property
    def cut_points(self):
        return self.cut_points_
