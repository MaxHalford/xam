"""
Minimum Description Length Principle (MDLP)

- Original paper: http://sci2s.ugr.es/keel/pdf/algorithm/congreso/fayyad1993.pdf
- Implementation inspiration: https://www.ibm.com/support/knowledgecenter/it/SSLVMB_21.0.0/com.ibm.spss.statistics.help/alg_optimal-binning.htm
"""

import collections

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_X_y


def calc_class_entropy(y):
    class_counts = stats.itemfreq(y)[:, 1]
    return stats.entropy(class_counts)


def calc_class_information_entropy(x, y, cut_point):
    partition = x < cut_point

    y_1 = y[partition]
    y_2 = y[-partition]

    ent_1 = calc_class_entropy(y_1)
    ent_2 = calc_class_entropy(y_2)

    return (y_1.size * ent_1 + y_2.size * ent_2) / (y_1.size + y_2.size)


class MDLPBinner(BaseEstimator, TransformerMixin):

    def __init__(self):
        # Attributes
        self.cut_points_ = None

    def fit(self, X, y):
        """Determine which are the best cut points for each column in X based on y."""

        # Check that X and y have correct shapes
        X, y = check_X_y(X, y)

        self.cut_points_ = [self.cut(x, y, []) for x in X.T]
        return self

    def transform(self, X, y=None):
        """Binarize X based on the fitted cut points."""
        X_discrete = np.array([
            np.digitize(X[:, i], self.cut_points_[i])
            for i in range(X.shape[1])
        ])
        return X_discrete

    def cut(self, x, y, cut_points):

        # No cut is necessary if there is only one class
        if np.unique(y).size == 1:
            return

        # Sort x and y according to x
        sorted_indexes = x.argsort()
        x = x[sorted_indexes]
        y = y[sorted_indexes]

        # Find the potential cut points
        potential_cut_points = []
        for i in range(x.size - 1):
            if x[i] != x[i+1] and y[i] != y[i+1]:
                potential_cut_points.append(x[i])

        # Ignore the cut points that appear more than once
        counts = collections.Counter(potential_cut_points)
        for cut_point in set(potential_cut_points):
            if counts[cut_point] > 1:
                potential_cut_points.remove[cut_point]

        # Find the cut point with gives the lowest class information entropy
        cut_point = min(
            potential_cut_points,
            key=lambda cut_point: calc_class_information_entropy(x, y, cut_point)
        )

        # Partition the data
        partition = x < cut_point
        x_1 = x[partition]
        y_1 = y[partition]
        x_2 = x[-partition]
        y_2 = y[-partition]

        # Get the number of unique classes in each group
        k = np.unique(y).size
        k_1 = np.unique(y_1).size
        k_2 = np.unique(y_2).size

        # Calculate the entropy of each group
        y_ent = calc_class_entropy(y)
        y_1_ent = calc_class_entropy(y_1)
        y_2_ent = calc_class_entropy(y_2)

        # Calculate the acceptance criterion
        delta = np.log2(3 ** k - 2) - k * y_ent + k_1 * y_1_ent + k_2 * y_2_ent
        n = y.size
        acceptance_criterion = (np.log2(n - 1) + delta) / n

        # Calculate the information gain obtained with the obtained cut point
        new_ent = calc_class_information_entropy(x, y, cut_point)
        gain = y_ent - new_ent

        # Add the cut point if the gain is higher than the acceptance criterion
        if gain > acceptance_criterion:
            cut_points.append(cut_point)
            # Recursively check if further cuts are possible
            self.cut(x_1, y_1, cut_points)
            self.cut(x_2, y_2, cut_points)

        return cut_points
