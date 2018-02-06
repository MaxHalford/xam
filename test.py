import pandas as pd
from sklearn import metrics
import xam

series = pd.Series([1, 2, 3, 4, 3, 4, 5, 4, 5, 6])
ewm = xam.tsa.calc_optimized_ewm(series, metric=metrics.mean_squared_error, seed=42)
print(ewm)
