"""
Equal frequency binning
"""

import numpy as np
from sklearn.utils import check_array

from .base import BaseUnsupervisedBinner


class EqualFrequencyBinner(BaseUnsupervisedBinner):

    def __init__(self, n_bins=5):

        super().__init__()

        # Properties
        self.n_bins = n_bins

    def fit(self, X, y=None, **fit_params):
        """Choose equally spaces cut points."""

        # scikit-learn checks
        X = check_array(X)

        p_step = 100 / self.n_bins

        self.cut_points_ = np.stack([
            np.percentile(X, p, axis=0)
            for p in np.arange(start=p_step, stop=100, step=p_step)
        ], axis=-1).tolist()

        return self
