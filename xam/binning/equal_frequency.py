"""
Equal frequency binning
"""

import numpy as np

from .base import BaseBinner


class EqualFrequencyBinner(BaseBinner):

    def __init__(self, n_bins):

        super().__init__()

        # Properties
        self.n_bins = n_bins

    def fit(self, X, y):
        """Choose equally spaces cut points."""

        p_step = 100 / self.n_bins

        self.cut_points_ = np.stack([
            np.percentile(X, p, axis=0)
            for p in np.arange(start=p_step, stop=100, step=p_step)
        ], axis=-1).tolist()

        return self
