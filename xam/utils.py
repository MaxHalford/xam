from collections import defaultdict
import codecs
import datetime as dt


def normalized_compression_distance(x, y, n=25_270_000_000):
    """Computes the Normalized Compression Distance (NCD) between two strings.

    Parameters:
        x (bytes)
        y (bytes)

    References:

        1. https://www.wikiwand.com/en/Normalized_compression_distance
        2. https://www.wikiwand.com/en/Normalized_Google_distance

    """

    x_code = codecs.encode(x, encoding='zip')
    y_code = codecs.encode(y, encoding='zip')
    x_y_code = codecs.encode(x + y, encoding='zip')

    return (len(x_y_code) - min(len(x_code), len(y_code))) / max(len(x_code), len(y_code))


def datetime_range(since, until, step=dt.timedelta(days=1)):
    """Generates datetimes in range [since, until] with a given step.

    Args:
        since (datetime)
        until (datetime)
        step (timedelta)

    Returns:
        a generator of datetimes
    """
    for i in range((until - since) // step + 1):
        yield since + step * i


def get_next_weekday(date, weekday):
    """Find the first weekday after a given date.

    Args:
        date (datetime)
        weekday (int)

    Returns:
        datetime: The next day of the week that has index `weekday`.
    """
    return date + dt.timedelta(days=(weekday - date.weekday() + 7) % 7)


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


def dataframe_to_vw(dataframe, label_col, importance_col=None, base_col=None, tag_col=None):
    """Convert a pandas.DataFrame to a string which follows Vowpal Wabbit's
    input format.

    Reference: https://github.com/JohnLangford/vowpal_wabbit/wiki/Input-format
    """

    vw_str = ''
    cols = dataframe.columns.tolist()

    label_idx = cols.index(label_col)
    importance_idx = cols.index(importance_col) if importance_col is not None else None
    base_idx = cols.index(base_col) if base_col is not None else None
    tag_idx = cols.index(tag_col) if tag_col is not None else None

    # Determine the columns that represent features
    feature_idxs = set((
        i for i, col in enumerate(cols)
        if col not in (label_col, importance_col, base_col, tag_col)
    ))

    for row in dataframe.itertuples(index=False):

        row_str = str(row[label_idx])

        if importance_idx:
            row_str += ' {}'.format(row[importance_idx])

        if base_idx:
            row_str += ' {}'.format(row[base_idx])

        if tag_idx:
            row_str += " '{}".format(row[tag_idx])

        row_str += ' | '
        row_str += ' '.join(('{}:{}'.format(cols[i], str(row[i])) for i in feature_idxs))

        vw_str += '{}\n'.format(row_str)

    # Remove the last carriage return
    vw_str = vw_str.rstrip('\n')

    return vw_str
