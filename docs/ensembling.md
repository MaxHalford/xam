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

## LightGBM with CV

```python
>>> import lightgbm as lgbm
>>> import numpy as np
>>> from sklearn import datasets, metrics, model_selection
>>> import xam

>>> X, y = datasets.load_breast_cancer(return_X_y=True)

>>> n_splits = 5
>>> cv = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)

>>> params = {
...     'objective': 'binary',
...     'metric': 'auc',
...     'verbosity': -1
... }

>>> single_scores = np.zeros(n_splits)
>>> with_cv_scores = np.zeros(n_splits)

>>> for i, (fit_idx, val_idx) in enumerate(cv.split(X, y)):
...
...     X_fit = X[fit_idx]
...     y_fit = y[fit_idx]
...     X_val = X[val_idx]
...     y_val = y[val_idx]
...
...     fit_set = lgbm.Dataset(X_fit, np.log1p(y_fit))
...     val_set = lgbm.Dataset(X_val, np.log1p(y_val))
...
...     # Train a single LGBM
...     model = lgbm.train(
...         params=params,
...         train_set=fit_set,
...         valid_sets=(fit_set, val_set),
...         num_boost_round=30,
...         verbose_eval=False,
...         early_stopping_rounds=5,
...     )
...     single_scores[i] = metrics.roc_auc_score(y_val, model.predict(X_val))
...
...     # Train a LGBM CV
...     model = xam.ensemble.LGBMCV(
...         cv=model_selection.KFold(3, shuffle=True, random_state=42),
...         **params
...     )
...     model = model.fit(
...         X_fit, y_fit,
...         num_boost_round=30,
...         verbose_eval=False,
...         early_stopping_rounds=5,
...     )
...     with_cv_scores[i] = metrics.roc_auc_score(y_val, model.predict(X_val))

>>> print('LGBM without CV AUC: {:.5f} (+/- {:.5f})'.format(single_scores.mean(), single_scores.std()))
LGBM without CV AUC: 0.98049 (+/- 0.01140)

>>> print('LGBM with CV AUC: {:.5f} (+/- {:.5f})'.format(with_cv_scores.mean(), with_cv_scores.std()))
LGBM with CV AUC: 0.98714 (+/- 0.00996)

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
...     'Random forest': RandomForestClassifier(n_estimators=10, random_state=1),
...     'Naïve Bayes': GaussianNB(),
...     'LightGBM': lgbm.LGBMClassifier(random_state=42, verbose=-1)
... }

>>> stack = xam.ensemble.StackingClassifier(
...     models=models,
...     meta_model=LogisticRegression(solver='lbfgs', multi_class='auto'),
...     metric=metrics.accuracy_score,
...     use_base_features=True,
...     use_probas=True,
... )

>>> for name, model in dict(models, **{'Stacking': stack}).items():
...     scores = model_selection.cross_val_score(model, X, y, cv=3, scoring='accuracy')
...     print('Accuracy: %0.3f (+/- %0.3f) [%s]' % (scores.mean(), 1.96 * scores.std(), name))
Accuracy: 0.913 (+/- 0.016) [KNN]
Accuracy: 0.934 (+/- 0.100) [Random forest]
Accuracy: 0.921 (+/- 0.052) [Naïve Bayes]
Accuracy: 0.934 (+/- 0.046) [LightGBM]
Accuracy: 0.954 (+/- 0.047) [Stacking]

```

### Stacking regression

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
...     meta_model=RandomForestRegressor(n_estimators=10, random_state=1),
...     cv=model_selection.KFold(n_splits=10),
...     use_base_features=True
... )

>>> for name, model in dict(models, **{'Stacking': stack}).items():
...     scores = model_selection.cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
...     print('MAE: %0.3f (+/- %0.3f) [%s]' % (-scores.mean(), 1.96 * scores.std(), name))
MAE: 7.338 (+/- 1.423) [KNN]
MAE: 4.250 (+/- 1.919) [Linear regression]
MAE: 4.112 (+/- 1.967) [Ridge regression]
MAE: 3.227 (+/- 1.065) [Stacking]

```


## Stacking with bagged test predictions

Averaging the predictions of each level 1 model instance means that we don't have to train a model on the full dataset. The meta-model is still trained on the out-of-fold predictions of the level 1 instances.

### Bagged stacking classification

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
...     'Random forest': RandomForestClassifier(n_estimators=10, random_state=1),
...     'Naïve Bayes': GaussianNB(),
...     'LightGBM': lgbm.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
... }

>>> stack = xam.ensemble.BaggedStackingClassifier(
...     models=models,
...     meta_model=RandomForestClassifier(n_estimators=100, random_state=1),
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
...     scores = model_selection.cross_val_score(model, X, y, cv=3, scoring='accuracy', error_score='raise')
...     print('Accuracy: %0.3f (+/- %0.3f) [%s]' % (scores.mean(), 1.96 * scores.std(), name))
Accuracy: 0.913 (+/- 0.016) [KNN]
Accuracy: 0.934 (+/- 0.100) [Random forest]
Accuracy: 0.921 (+/- 0.052) [Naïve Bayes]
Accuracy: 0.940 (+/- 0.053) [LightGBM]
Accuracy: 0.967 (+/- 0.048) [Stacking]

```

### Bagged stacking regression

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

>>> stack = xam.ensemble.BaggedStackingRegressor(
...     models=models,
...     meta_model=RandomForestRegressor(n_estimators=10, random_state=1),
...     cv=model_selection.KFold(n_splits=10),
...     use_base_features=True
... )

>>> for name, model in dict(models, **{'Stacking': stack}).items():
...     scores = model_selection.cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
...     print('MAE: %0.3f (+/- %0.3f) [%s]' % (-scores.mean(), 1.96 * scores.std(), name))
MAE: 7.338 (+/- 1.423) [KNN]
MAE: 4.250 (+/- 1.919) [Linear regression]
MAE: 4.112 (+/- 1.967) [Ridge regression]
MAE: 3.211 (+/- 1.100) [Stacking]


```
