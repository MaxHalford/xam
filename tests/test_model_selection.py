import datetime as dt

import numpy as np
import pandas as pd

import xam


def test_ordered_cross_validation():

    # Test with integers

    X = pd.DataFrame(index=[1, 2, 3, 4])
    n_splits = 2
    delta = 1
    splits = [
        [[0, 1, 2], [3]],
        [[0, 1], [2]]
    ]

    cv = xam.model_selection.OrderedCV(n_splits=n_splits, delta=delta)

    for i, (train_idx, test_idx) in enumerate(cv.split(X)):
        np.testing.assert_array_equal(train_idx, splits[i][0])
        np.testing.assert_array_equal(test_idx, splits[i][1])

    # Test with dates

    date = dt.datetime

    X = pd.DataFrame(index=[
        date(2017, 1, 1),
        date(2017, 1, 2),
        date(2017, 1, 4),
        date(2017, 1, 5),
        date(2017, 1, 7),
        date(2017, 1, 8),
        date(2017, 1, 9),
    ])
    n_splits = 3
    delta = dt.timedelta(days=2)
    splits = [
        [[0, 1, 2, 3, 4], [5, 6]],
        [[0, 1, 2, 3], [4]],
        [[0, 1], [2, 3]],
    ]

    cv = xam.model_selection.OrderedCV(n_splits=n_splits, delta=delta)

    for i, (train_idx, test_idx) in enumerate(cv.split(X)):
        np.testing.assert_array_equal(train_idx, splits[i][0])
        np.testing.assert_array_equal(test_idx, splits[i][1])
