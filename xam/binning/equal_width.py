"""
Equal width binning
"""

import numpy as np

from .base import BaseUnsupervisedBinner


class EqualWidthBinner(BaseUnsupervisedBinner):

    def __init__(self, n_bins):

        super().__init__()

        # Properties
        self.n_bins = n_bins

    def fit(self, X, y=None):
        """Choose equally spaces cut points."""

        self.cut_points_ = [0] * X.shape[1]

        for i, x in enumerate(X.T):
            x_min = np.min(x)
            x_max = np.max(x)
            step = (x_max - x_min) / self.n_bins
            self.cut_points_[i] = np.arange(start=x_min+step, stop=x_max, step=step).tolist()

        return self
