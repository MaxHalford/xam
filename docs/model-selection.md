# Model selection

## Datetime cross-validation

```python
>>> import datetime as dt
>>> import pandas as pd
>>> import xam

>>> df = pd.DataFrame(
...     [1, 2, 3, 4, 5, 6],
...     index=[
...         dt.datetime(2016, 5, 1),
...         dt.datetime(2016, 5, 1),
...         dt.datetime(2016, 5, 2),
...         dt.datetime(2016, 5, 2),
...         dt.datetime(2016, 5, 2),
...         dt.datetime(2016, 5, 3),
...     ]
... )

>>> df
            0
2016-05-01  1
2016-05-01  2
2016-05-02  3
2016-05-02  4
2016-05-02  5
2016-05-03  6

>>> cv = xam.model_selection.DatetimeCV(timedelta=dt.timedelta(days=1))

>>> for train_idxs, test_idxs in cv.split(df):
...     print(train_idxs, test_idxs)
[0 1] [2 3 4]
[0 1 2 3 4] [5]

```
