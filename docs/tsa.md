# Time series analysis (TSA)

## Exponentially weighted average optimization

The `calc_optimized_ewm` tries to find the `alpha` that minimises a `metric` between a series and the [exponentially weighted average (EWM)](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.ewm.html) of parameter `alpha`. The optimisation is done with [differential evolution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html). The EWM is calculated on the input series shifted by an input paramater called `shift`. This ensures that the EWM can only look at values that are in the past with a margin of `shift`. For example if you want the EWM to predict today's values with data that only goes up to yesterday then you would set `shift` to be equal to 1 (which also happens to be the default value).

```python
>>> import pandas as pd
>>> from sklearn import metrics
>>> import xam

>>> series = pd.Series([1, 2, 3, 1, 2, 3, 1, 2, 3])
>>> shift = 3
>>> metric = metrics.mean_squared_error
>>> ewm = xam.tsa.calc_optimized_ewm(series, shift=shift, metric=metric, seed=42)
>>> ewm
0       NaN
1       NaN
2       NaN
3    1.0000
4    1.9999
5    2.9999
6    1.0002
7    1.9999
8    2.9999
dtype: float64

>>> metric(series[shift:], ewm[shift:])
1.3351909851905068e-08

```

## Exponential smoothing

```python
>>> import datetime as dt
>>> from math import sqrt
>>> import numpy as np
>>> import pandas as pd
>>> from sklearn import metrics
>>> import xam

>>> df = pd.read_csv('datasets/airline-passengers.csv')
>>> series = pd.Series(
...     data=df['passengers'].tolist(),
...     index=pd.DatetimeIndex([dt.datetime.strptime(m, '%Y-%m') for m in df['month']]),
...     dtype=np.float
... )

>>> # Determine how long a season lasts (in this case twelve months)
>>> season_length = 12

>>> # Train/test split
>>> train_test_split_index = 12 # Forecast the last year
>>> train = series[:-train_test_split_index]
>>> test = series[-train_test_split_index:]

>>> # Learning coefficients
>>> alpha = 0.1
>>> beta = 0.2
>>> gamma = 0.6

>>> pred = xam.tsa.SimpleExponentialSmoothingForecaster(alpha).fit(train).predict(test.index)
>>> print('RMSE: {:.3f}'.format(sqrt(metrics.mean_squared_error(test, pred))))
RMSE: 99.293

>>> pred = xam.tsa.DoubleExponentialSmoothingForecaster(alpha, beta).fit(train).predict(test.index)
>>> print('RMSE: {:.3f}'.format(sqrt(metrics.mean_squared_error(test, pred))))
RMSE: 73.265

>>> pred = xam.tsa.TripleExponentialSmoothingForecaster(
...     alpha,
...     beta,
...     gamma,
...     season_length=season_length,
...     multiplicative=True
... ).fit(train).predict(test.index)
>>> print('RMSE: {:.3f}'.format(sqrt(metrics.mean_squared_error(test, pred))))
RMSE: 17.543

```

## Frequency average forecasting

```python
>>> import datetime as dt
>>> import pandas as pd
>>> import xam

>>> df = pd.read_csv('datasets/bike-station.csv')
>>> series = pd.Series(
...     data=df['bikes'].tolist(),
...     index=pd.to_datetime(df['moment'], format='%Y-%m-%d %H:%M:%S')
... )

>>> forecaster = xam.tsa.FrequencyAverageForecaster(lambda d: f'{d.weekday()}-{d.hour}')
>>> forecaster.fit(series[:-10]).predict(series.index[-10:])
moment
2016-10-05 09:28:48    8.622535
2016-10-05 09:32:34    8.622535
2016-10-05 09:40:55    8.622535
2016-10-05 09:42:34    8.622535
2016-10-05 09:45:06    8.622535
2016-10-05 09:46:29    8.622535
2016-10-05 09:50:54    8.622535
2016-10-05 09:53:00    8.622535
2016-10-05 09:54:04    8.622535
2016-10-05 09:57:18    8.622535
dtype: float64

```
