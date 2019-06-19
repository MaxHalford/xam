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

`xam.utils.dataframe_to_vw` converts a `pandas.DataFrame` to a string which can be ingested by [Vowpal Wabbit](https://github.com/JohnLangford/vowpal_wabbit) once it is saved on disk.

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

## Skyline querying

`xam.utils.find_skyline` returns the [skyline](https://www.wikiwand.com/en/Skyline_operator) of a `pandas.DataFrame`.

```python
>>> import xam

>>> karts = pd.DataFrame(
...     data=[
...         ('Red Fire', 5, 4, 4, 5, 2),
...         ('Green Fire', 7, 3, 3, 4, 2),
...         ('Heart Coach', 4, 6, 6, 5, 2),
...         ('Bloom Coach', 6, 4, 5, 3, 2),
...         ('Turbo Yoshi', 4, 5, 6, 6, 2),
...         ('Turbo Birdo', 6, 4, 4, 7, 2),
...         ('Goo-Goo Buggy', 1, 9, 9, 2, 3),
...         ('Rattle Buggy', 2, 9, 8, 2, 3),
...         ('Toad ', 3, 9, 7, 2, 3),
...         ('Toadette ', 1, 9, 9, 2, 3),
...         ('Koopa Dasher', 2, 8, 8, 3, 3),
...         ('Para-Wing', 1, 8, 9, 3, 3),
...         ('DK Jumbo', 8, 2, 2, 8, 1),
...         ('Barrel Train', 8, 7, 3, 5, 3),
...         ('Koopa King', 9, 1, 1, 9, 1),
...         ('Bullet Blaster', 8, 1, 4, 1, 3),
...         ('Wario Car', 7, 3, 3, 7, 1),
...         ('Waluigi Racer', 5, 9, 5, 6, 2),
...         ('Piranha Pipes', 8, 7, 2, 9, 1),
...         ('Boo Pipes', 2, 9, 8, 9, 1),
...         ('Parade Kart', 7, 3, 4, 7, 3)
...     ],
...     columns=['name', 'speed', 'off_road', 'acceleration', 'weight', 'turbo']
... )

>>> skyline = xam.utils.find_skyline(
...     df=karts,
...     to_max=['speed', 'off_road', 'acceleration', 'turbo'],
...     to_min=['weight']
... )

>>> skyline
              name  speed  off_road  acceleration  weight  turbo
1       Green Fire      7         3             3       4      2
2      Heart Coach      4         6             6       5      2
3      Bloom Coach      6         4             5       3      2
6    Goo-Goo Buggy      1         9             9       2      3
7     Rattle Buggy      2         9             8       2      3
8            Toad       3         9             7       2      3
9        Toadette       1         9             9       2      3
13    Barrel Train      8         7             3       5      3
14      Koopa King      9         1             1       9      1
15  Bullet Blaster      8         1             4       1      3
17   Waluigi Racer      5         9             5       6      2
20     Parade Kart      7         3             4       7      3

```
