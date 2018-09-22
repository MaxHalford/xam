# Ensembling

## Groupby model

```python
>>> import pandas as pd
>>> from sklearn import model_selection
>>> from sklearn.linear_model import Lasso
>>> from sklearn.datasets import load_diabetes
>>> import xam

>>> X, y = load_diabetes(return_X_y=True)
>>> X = pd.DataFrame(X)
>>> X['split'] = X[[3]] > X[[3]].mean()

>>> lasso = Lasso(alpha=0.01, random_state=42)
>>> split_lasso = xam.ensemble.GroupbyModel(lasso, 'split')

>>> cv = model_selection.KFold(n_splits=5, random_state=42)

>>> scores = model_selection.cross_val_score(lasso, X, y, cv=cv, scoring='neg_mean_squared_error')
>>> print('{:.3f} (+/- {:.3f})'.format(-scores.mean(), 1.96 * scores.std()))
3016.066 (+/- 244.806)

>>> scores = model_selection.cross_val_score(split_lasso, X, y, cv=cv, scoring='neg_mean_squared_error')
>>> print('{:.3f} (+/- {:.3f})'.format(-scores.mean(), 1.96 * scores.std()))
2902.065 (+/- 265.931)

```

## Stacking

- [A Kaggler's guide to model stacking in practice](http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/)
- [Stacking Made Easy: An Introduction to StackNet by Competitions Grandmaster Marios Michailidis](http://blog.kaggle.com/2017/06/15/stacking-made-easy-an-introduction-to-stacknet-by-competitions-grandmaster-marios-michailidis-kazanova/)
- [Stacked generalization, when does it work?](http://www.cs.waikato.ac.nz/~ihw/papers/97KMT-IHW-Stacked.pdf)

### Stacking classification

```python
>>> import lightgbm as lgbm
>>> from sklearn import datasets, metrics, model_selection
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.naive_bayes import GaussianNB
>>> from sklearn.neighbors import KNeighborsClassifier
>>> import xam

>>> iris = datasets.load_iris()
>>> X, y = iris.data[:, 1:3], iris.target

>>> models = {
...     'KNN': KNeighborsClassifier(n_neighbors=1),
...     'Random forest': RandomForestClassifier(random_state=1),
...     'Naïve Bayes': GaussianNB(),
...     'LightGBM': lgbm.LGBMClassifier(random_state=42, verbose=-1)
... }

>>> stack = xam.ensemble.StackingClassifier(
...     models=models,
...     meta_model=LogisticRegression(),
...     metric=metrics.accuracy_score,
...     use_base_features=True,
...     use_probas=True,
...     fit_handlers={
...         'LightGBM': lambda X_fit, y_fit, X_val, y_val: {
...             'eval_set': [(X_fit, y_fit), (X_val, y_val)],
...             'eval_names': ['fit', 'val'],
...             'early_stopping_rounds': 10,
...             'verbose': False
...         }
...     }
... )

>>> for name, model in dict(models, **{'Stacking': stack}).items():
...     scores = model_selection.cross_val_score(model, X, y, cv=3, scoring='accuracy')
...     print('Accuracy: %0.3f (+/- %0.3f) [%s]' % (scores.mean(), 1.96 * scores.std(), name))
Accuracy: 0.913 (+/- 0.016) [KNN]
Accuracy: 0.914 (+/- 0.126) [Random forest]
Accuracy: 0.921 (+/- 0.052) [Naïve Bayes]
Accuracy: 0.934 (+/- 0.046) [LightGBM]
Accuracy: 0.954 (+/- 0.047) [Stacking]

```

**Stacking regression**

Model stacking for regression as described in this [Kaggle blog post](http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/).

```python
>>> from sklearn import datasets, model_selection
>>> from sklearn.ensemble import RandomForestRegressor
>>> from sklearn.linear_model import LinearRegression
>>> from sklearn.linear_model import Ridge
>>> from sklearn.neighbors import KNeighborsRegressor
>>> import xam

>>> boston = datasets.load_boston()
>>> X, y = boston.data, boston.target

>>> models = {
...     'KNN': KNeighborsRegressor(n_neighbors=1),
...     'Linear regression': LinearRegression(),
...     'Ridge regression': Ridge(alpha=.5)
... }

>>> stack = xam.ensemble.StackingRegressor(
...     models=models,
...     meta_model=RandomForestRegressor(random_state=1),
...     cv=model_selection.KFold(n_splits=10),
...     use_base_features=True
... )

>>> for name, model in dict(models, **{'Stacking': stack}).items():
...     scores = model_selection.cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
...     print('MAE: %0.3f (+/- %0.3f) [%s]' % (-scores.mean(), 1.96 * scores.std(), name))
MAE: 7.338 (+/- 1.423) [KNN]
MAE: 4.257 (+/- 1.923) [Linear regression]
MAE: 4.118 (+/- 1.971) [Ridge regression]
MAE: 3.234 (+/- 1.089) [Stacking]

```
