import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

from xam.tsa import FrequencyAverageForecaster
from xam.tsa import util


df = pd.read_csv('data/bike-station.csv')

series = pd.Series(
    data=df['bikes'].tolist(),
    index=pd.to_datetime(df['moment'], format='%Y-%m-%d %H:%M:%S')
)

train_until = series.index.max() - dt.timedelta(days=7)

forecaster = FrequencyAverageForecaster(lambda d: f'{d.weekday()}-{d.hour}')

score = util.temporal_train_test_score(
    forecaster=forecaster,
    series=series,
    train_until=train_until,
    metric=metrics.mean_squared_error
)

print('MSE: %f' % score)
