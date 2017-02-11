"""
Equal frequency binning
"""

import numpy as np

from .base import BaseUnsupervisedBinner
from ..base import Model
from ..check import is_a_positive_int


class EqualFrequencyBinner(BaseUnsupervisedBinner, Model):

    def __init__(self, n_bins):

        super().__init__()

        # Properties
        self.n_bins = n_bins

    def fit(self, X, y=None):
        """Choose equally spaces cut points."""

        p_step = 100 / self.n_bins

        self.cut_points_ = np.stack([
            np.percentile(X, p, axis=0)
            for p in np.arange(start=p_step, stop=100, step=p_step)
        ], axis=-1).tolist()

        return self

    def check_params(self):
        # Check n_bins is a positive int
        if not is_a_positive_int(self.n_bins):
            raise ValueError('n_bins is not a strictly positive int')
        return
