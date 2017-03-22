from collections import defaultdict
import datetime as dt
from itertools import groupby
from operator import itemgetter

import numpy as np


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


def intraclass_correlation(x, y):
    """Calculate the correlation ratio between a categorical array and a numerical one.

    Args:
        x: A sequence of numbers.
        y: A sequence of strings.

    Returns:
        float: The correlation ratio between x and y.
    """
    groups = groupby(sorted(zip(y, x), key=itemgetter(0)), key=itemgetter(0))
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


def subsequence_lengths(sequence):
    """Calculate the lengths of each subsequence in a sequence.

    Args:
        sequence (iterable): 'abbaabbbb'
    Returns:
        dict: {'a': [1, 2], 'b': [2, 4]}
    """

    lengths = defaultdict(list)

    # Go through the first n-1 elements
    i = 1
    for pre, post in zip(sequence, sequence[1:]):
        if pre == post:
            i += 1
        else:
            lengths[pre].append(i)
            i = 1

    # Check the nth element
    if sequence[-1] == sequence[-2]:
        lengths[sequence[-1]].append(i)
    else:
        lengths[sequence[-2]].append(i+1)
        lengths[sequence[-1]].append(1)

    return dict(lengths)
