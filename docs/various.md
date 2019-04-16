# Various

## Datetime range

```python
>>> import datetime as dt
>>> import xam

>>> since = dt.datetime(2017, 3, 22)
>>> until = dt.datetime(2017, 3, 25)
>>> step = dt.timedelta(days=2)

>>> dt_range = xam.utils.datetime_range(since=since, until=until, step=step)
>>> for dt in dt_range:
...     print(dt)
2017-03-22 00:00:00
2017-03-24 00:00:00

```

## Next day of the week

```python
>>> import datetime as dt
>>> import xam

>>> now = dt.datetime(2017, 3, 22) # Wednesday
>>> next_monday = xam.utils.get_next_weekday(now, 0) # Get next Monday
>>> next_monday
datetime.datetime(2017, 3, 27, 0, 0)

```

## Subsequence lengths

```python
>>> import xam

>>> sequence = 'appaaaaapa'
>>> lengths = xam.utils.subsequence_lengths(sequence)
>>> print(lengths)
{'a': [1, 5, 1], 'p': [2, 1, 2]}

>>> averages = {k: sum(v) / len(v) for k, v in lengths.items()}
>>> print(averages)
{'a': 2.3333333333333335, 'p': 1.6666666666666667}

```

## DataFrame to Vowpal Wabbit

`xam.utils.dataframe_to_vw` convert a `pandas.DataFrame` to a string which can be ingested by [Vowpal Wabbit](https://github.com/JohnLangford/vowpal_wabbit) once it is saved on disk.

```python
>>> import pandas as pd
>>> import xam

>>> df = pd.DataFrame.from_dict({
...     'label': [0, 0, 1, 1],
...     'feature_0': [0.2, 0.1, 0.4, 0.3],
...     'feature_1': [0.4, 0.3, 0.3, 0.2],
... })

>>> vw_str = xam.utils.dataframe_to_vw(df, label_col='label')
>>> print(vw_str)
0 | feature_0:0.2 feature_1:0.4
0 | feature_0:0.1 feature_1:0.3
1 | feature_0:0.4 feature_1:0.3
1 | feature_0:0.3 feature_1:0.2

```

## Normalized compression distance

```python
>>> import xam

>>> x = b'A blue cow'
>>> y = b'The green goat'
>>> xam.utils.normalized_compression_distance(x, y)
0.6363636363636364

```
