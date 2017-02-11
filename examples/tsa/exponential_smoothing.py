import datetime as dt
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xam.tsa import DoubleExponentialSmoothingForecaster
from xam.tsa import SimpleExponentialSmoothingForecaster
from xam.tsa import TripleExponentialSmoothingForecaster


df = pd.read_csv('data/airline-passengers.csv')

series = pd.Series(
    data=df['passengers'].tolist(),
    index=pd.DatetimeIndex([dt.datetime.strptime(m, '%Y-%m') for m in  df['month']]),
    dtype=np.float
)

# Train/test split

season_length = 12
train_test_split_index = 100
train = series[:train_test_split_index]
test = series[train_test_split_index:]

# Fitting

simple = SimpleExponentialSmoothingForecaster(alpha=0.3)
double = DoubleExponentialSmoothingForecaster(alpha=0.3, beta=0.3)
triple = TripleExponentialSmoothingForecaster(
    alpha=0.1,
    beta=0.2,
    gamma=0.6,
    season_length=season_length,
    multiplicative=True
)

simple.fit(train)
double.fit(train)
triple.fit(train)

# Plotting

fig, ax = plt.subplots()

series.plot(ax=ax, label='Actual', alpha=0.6)

simple.fitted_.append(simple.predict(test.index)).plot(
    ax=ax,
    label='Simple',
    alpha=0.6,
    linestyle='--'
)

double.fitted_.append(double.predict(test.index)).plot(
    ax=ax,
    label='Double',
    alpha=0.6,
    linestyle='--'
)

triple.fitted_.append(triple.predict(test.index)).plot(
    ax=ax,
    label='Triple',
    alpha=0.6,
    linestyle='--'
)

ax.set_ylim((0, 1000))

ax.axvspan(
    series.index[0],
    series.index[train_test_split_index],
    facecolor='gray',
    alpha=0.3,
    label='Training'
)

plt.legend(loc='upper left')
plt.show()
