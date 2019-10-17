from collections import defaultdict
import codecs
import datetime as dt

import pandas as pd
import numpy as np


__all__ = [
    'dataframe_to_vw',
    'datetime_range',
    'find_partitions',
    'find_skyline',
    'get_next_weekday',
    'normalized_compression_distance',
    'subsequence_lengths'
]


def find_skyline(df, to_min, to_max):
    """Finds the skyline of a dataframe using a block-nested loop algorithm."""

    def count_diffs(a, b, to_min, to_max):
        n_better = 0
        n_worse = 0

        for f in to_min:
            n_better += a[f] < b[f]
            n_worse += a[f] > b[f]

        for f in to_max:
            n_better += a[f] > b[f]
            n_worse += a[f] < b[f]

        return n_better, n_worse

    rows = df.to_dict(orient='index')

    # Use the first row to initialize the skyline
    skyline = {df.index[0]}

    # Loop through the rest of the rows
    for i in df.index[1:]:

        to_drop = set()
        is_dominated = False

        for j in skyline:

            n_better, n_worse = count_diffs(rows[i], rows[j], to_min, to_max)

            # Case 1
            if n_worse > 0 and n_better == 0:
                is_dominated = True
                break

            # Case 3
            if n_better > 0 and n_worse == 0:
                to_drop.add(j)

        if is_dominated:
            continue

        skyline = skyline.difference(to_drop)
        skyline.add(i)

    return df[df.index.isin(skyline)]


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
        lengths[sequence[-2]].append(i + 1)
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


def find_partitions(df, match_func, max_size=None, block_by=None):
    """Recursive algorithm for finding duplicates in a DataFrame."""

    # If block_by is provided, then we apply the algorithm to each block and
    # stitch the results back together
    if block_by is not None:
        blocks = df.groupby(block_by).apply(lambda g: find_partitions(
            df=g,
            match_func=match_func,
            max_size=max_size
        ))

        keys = blocks.index.unique(block_by)
        for a, b in zip(keys[:-1], keys[1:]):
            blocks.loc[b, :] += blocks.loc[a].iloc[-1] + 1

        return blocks.reset_index(block_by, drop=True)

    def get_record_index(r):
        return r[df.index.name or 'index']

    # Records are easier to work with than a DataFrame
    records = df.to_records()

    # This is where we store each partition
    partitions = []

    def find_partition(at=0, partition=None, indexes=None):

        r1 = records[at]

        if partition is None:
            partition = {get_record_index(r1)}
            indexes = [at]

        # Stop if enough duplicates have been found
        if max_size is not None and len(partition) == max_size:
            return partition, indexes

        for i, r2 in enumerate(records):

            if get_record_index(r2) in partition or i == at:
                continue

            if match_func(r1, r2):
                partition.add(get_record_index(r2))
                indexes.append(i)
                find_partition(at=i, partition=partition, indexes=indexes)

        return partition, indexes

    while len(records) > 0:
        partition, indexes = find_partition()
        partitions.append(partition)
        records = np.delete(records, indexes)

    return pd.Series({
        idx: partition_id
        for partition_id, idxs in enumerate(partitions)
        for idx in idxs
    })
