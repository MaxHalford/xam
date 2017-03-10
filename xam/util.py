import datetime as dt
from itertools import groupby
from operator import itemgetter

import numpy as np


def correlation_ratio(x, y):
    """Calculate the correlation ratio between a categorical array and a numerical one.

    Args:
        x: A sequence of strings.
        y: A sequence of numbers.

    Returns:
        float: The correlation ratio between x and y.
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


def datetime_range(since, until, step=dt.timedelta(days=1)):
    """Generates datetimes in range [since, until] with a given step.

    Args:
        since (datetime)
        until (datetime)
        step (timedelta)

    Returns:
        list of datetimes
    """
    n_steps = (until - since) // step
    return [
        since + step * i
        for i in range(n_steps + 1)
    ]
