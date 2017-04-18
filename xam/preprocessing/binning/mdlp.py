"""
Minimum Description Length Principle (MDLP) binning

- Original paper: http://sci2s.ugr.es/keel/pdf/algorithm/congreso/fayyad1993.pdf
- Implementation inspiration: https://www.ibm.com/support/knowledgecenter/it/SSLVMB_21.0.0/com.ibm.spss.statistics.help/alg_optimal-binning.htm
"""

import collections
import math

import numpy as np
from scipy import stats
from sklearn.utils import as_float_array
from sklearn.utils import check_X_y

from .base import BaseSupervisedBinner


class MDLPBinner(BaseSupervisedBinner):

    def fit(self, X, y, **fit_params):
        """Determine which are the best cut points for each column in X based on y."""

        X = as_float_array(X)
        y = np.array(y).astype(int)
        X, y = check_X_y(X, y)

        self.cut_points_ = [mdlp_cut(x, y, []) for x in X.T]
        return self


def calc_class_entropy(y):
    class_counts = stats.itemfreq(y)[:, 1]
    return stats.entropy(class_counts, base=2)


def calc_class_information_entropy(x, y, cut_point):
    partition = x <= cut_point

    y_1 = y[partition]
    y_2 = y[~partition]

    ent_1 = calc_class_entropy(y_1)
    ent_2 = calc_class_entropy(y_2)

    return (y_1.size * ent_1 + y_2.size * ent_2) / (y_1.size + y_2.size)


def mdlp_cut(x, y, cut_points):

    # No cut is necessary if there is only one class
    if len(np.unique(y)) == 1:
        return

    # Calculate the current entropy
    y_ent = calc_class_entropy(y)

    # Sort x and y according to x
    sorted_indexes = x.argsort()
    x = x[sorted_indexes]
    y = y[sorted_indexes]

    # Find the potential cut points
    potential_cut_points = []
    for i in range(x.size - 1):
        potential_cut_points.append((x[i] + x[i+1]) / 2)

    # Ignore the cut points that appear more than once
    potential_cut_points = list(set(potential_cut_points))

    # Find the cut point with gives the lowest class information entropy
    cut_point = min(
        potential_cut_points,
        key=lambda cut_point: calc_class_information_entropy(x, y, cut_point)
    )

    # Calculate the information gain obtained with the obtained cut point
    new_ent = calc_class_information_entropy(x, y, cut_point)
    gain = y_ent - new_ent

    # Partition the data
    partition = x <= cut_point
    x_1 = x[partition]
    y_1 = y[partition]
    x_2 = x[~partition]
    y_2 = y[~partition]

    # Get the number of unique classes in each group
    k = len(np.unique(y))
    k_1 = len(np.unique(y_1))
    k_2 = len(np.unique(y_2))

    # Calculate the entropy of each group
    y_1_ent = calc_class_entropy(y_1)
    y_2_ent = calc_class_entropy(y_2)

    # Calculate the acceptance criterion
    delta = math.log2(3 ** k) - k * y_ent + k_1 * y_1_ent + k_2 * y_2_ent
    n = y.size
    acceptance_criterion = (math.log2(n - 1) + delta) / n

    # Add the cut point if the gain is higher than the acceptance criterion
    if gain > acceptance_criterion:
        cut_points.append(cut_point)
        # Recursively check if further cuts are possible
        mdlp_cut(x_1, y_1, cut_points)
        mdlp_cut(x_2, y_2, cut_points)

    return sorted(cut_points)
