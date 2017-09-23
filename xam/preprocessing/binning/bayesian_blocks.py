"""
Bayesian blocks binning. Good for visualization.

References:
    - http://adsabs.harvard.edu/abs/2012arXiv1207.5578S
    - https://jakevdp.github.io/blog/2012/09/12/dynamic-programming-in-python/
"""

import numpy as np
from sklearn.utils import check_array

from .base import BaseUnsupervisedBinner


class BayesianBlocksBinner(BaseUnsupervisedBinner):

    def fit(self, X, y=None, **fit_params):

        # scikit-learn checks
        X = check_array(X)

        self.cut_points_ = [calc_bayesian_blocks(x) for x in X.T]
        return self

    @property
    def cut_points(self):
        return self.cut_points_


def calc_bayesian_blocks(x):

    # Copy and sort the array
    x = np.sort(x)
    n = x.size

    # Create length-(n + 1) array of cell edges
    edges = np.concatenate([
        x[:1],
        0.5 * (x[1:] + x[:-1]),
        x[-1:]
    ])
    block_length = x[-1] - edges

    # Arrays needed for the iteration
    nn_vec = np.ones(n)
    best = np.zeros(n, dtype=float)
    last = np.zeros(n, dtype=int)

    # Start with first data cell; add one cell at each iteration
    for k in range(n):
        # Compute the width and count of the final bin for all possible
        # locations of the k^th changepoint
        width = block_length[:k + 1] - block_length[k + 1]
        count_vec = np.cumsum(nn_vec[:k + 1][::-1])[::-1]

        # Evaluate fitness function for these possibilities
        fit_vec = count_vec * (np.log(count_vec) - np.log(width))
        fit_vec -= 4  # 4 comes from the prior on the number of changepoints
        fit_vec[1:] += best[:k]

        # Find the max of the fitness: this is the k^th changepoint
        i_max = np.argmax(fit_vec)
        last[k] = i_max
        best[k] = fit_vec[i_max]

    # Recover changepoints by iteratively peeling off the last block
    change_points = np.zeros(n, dtype=int)
    i_cp = n
    ind = n
    while True:
        i_cp -= 1
        change_points[i_cp] = ind
        if ind == 0:
            break
        ind = last[ind - 1]
    change_points = change_points[i_cp:]

    return edges[change_points][1:-1]
