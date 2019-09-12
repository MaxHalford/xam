# Feature selection

## Forward-backward selection

```python
>>> import pandas as pd
>>> from sklearn import datasets
>>> from sklearn import model_selection
>>> from sklearn import pipeline
>>> from sklearn import neighbors
>>> from xam import feature_selection

>>> boston = datasets.load_boston()
>>> X = pd.DataFrame(boston.data, columns=boston.feature_names)
>>> y = boston.target

>>> def score(X, y):
...     return -model_selection.cross_val_score(
...         estimator=neighbors.KNeighborsRegressor(n_neighbors=3),
...         X=X,
...         y=y,
...         cv=model_selection.KFold(n_splits=5, shuffle=True, random_state=42),
...         scoring='neg_mean_squared_error'
...     ).mean()

>>> fbs = feature_selection.ForwardBackwardSelector(
...     score_func=score,
...     n_forward=2,
...     n_backward=1,
...     verbose=True
... )

>>> fbs = fbs.fit(X, y)
Added feature "LSTAT", new best score is 37.29318
Added feature "RM", new best score is 23.53988
Added feature "CRIM", new best score is 16.49524
Added feature "NOX", new best score is 16.43345
Added feature "CHAS", new best score is 16.43008
No further improvement was found, stopping

>>> fbs.transform(X).head()
   CHAS     CRIM  LSTAT    NOX     RM
0   0.0  0.00632   4.98  0.538  6.575
1   0.0  0.02731   9.14  0.469  6.421
2   0.0  0.02729   4.03  0.469  7.185
3   0.0  0.03237   2.94  0.458  6.998
4   0.0  0.06905   5.33  0.458  7.147

```
