from collections import defaultdict

from sklearn import metrics


def temporal_train_test_split(series, train_until):
    train_indexes = series.index <= train_until
    train = series[train_indexes]
    test = series[~train_indexes]
    return train, test


def temporal_train_test_score(forecaster, series, train_until, metric=metrics.mean_squared_error):
    train, test = temporal_train_test_split(series, train_until)
    forecaster.fit(train)
    pred = forecaster.predict(test.index)
    score = metric(test, pred)
    return score


def calc_subsequence_lengths(sequence):
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

    return lengths
