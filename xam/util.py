from itertools import groupby
from operator import itemgetter

import numpy as np


def correlation_ratio(x, y):
    """Calculate the correlation ratio between a categorical array and a numerical array one.
    Args:
        x (1D array_like): contains categorial data.
        y (1D array_like): contains numerical data.
    Returns:
        float: The correlation ratio between `A` and `B`.
    """
    groups = groupby(sorted(zip(x, y), key=itemgetter(0)), key=itemgetter(0))
    means = []
    variances = []
    sizes = []
    for _, group in groups:
        values = [v for _, v in group]
        means.append(np.mean(values))
        variances.append(np.var(values))
        sizes.append(len(values))

    # Total number of values, is also equal to the length of x and y
    n = sum(sizes)
    # Global mean
    mean = np.mean(means)
    # The inter-group variance is the weighted mean of the group means
    var_inter = sum((s * (m - mean) ** 2 for s, m in zip(sizes, means))) / n
    # The intra-group variance is the weighted variance of the group variances
    var_intra = sum((s * v for s, v in zip(sizes, variances))) / n
    return var_inter / (var_inter + var_intra)
