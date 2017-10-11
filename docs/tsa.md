# Time series analysis (TSA)

## Exponential smoothing forecasting

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

**Frequency average forecasting**

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
