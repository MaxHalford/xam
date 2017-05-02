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
