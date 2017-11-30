# Model selection

## Ordered cross-validation

```python
>>> import datetime as dt
>>> import pandas as pd
>>> import xam

>>> df = pd.DataFrame(
...     [1, 2, 3, 4, 5, 6, 7, 8],
...     index=[
...         dt.datetime(2016, 5, 1),
...         dt.datetime(2016, 5, 1),
...         dt.datetime(2016, 5, 2),
...         dt.datetime(2016, 5, 2),
...         dt.datetime(2016, 5, 3),
...         dt.datetime(2016, 5, 3),
...         dt.datetime(2016, 5, 3),
...         dt.datetime(2016, 5, 4),
...     ]
... )

>>> df
            0
2016-05-01  1
2016-05-01  2
2016-05-02  3
2016-05-02  4
2016-05-03  5
2016-05-03  6
2016-05-03  7
2016-05-04  8

>>> cv = xam.model_selection.OrderedCV(n_splits=2, delta=dt.timedelta(days=1))

>>> for train_idxs, test_idxs in cv.split(df):
...     print(train_idxs, test_idxs)
[0 1 2 3 4 5 6] [7]
[0 1 2 3] [4 5 6]

```
