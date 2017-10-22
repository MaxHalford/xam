# Linear models

## AUC regressor

This is the [AUC regressor](https://github.com/pyduan/amazonaccess/blob/f8addfefcee80f0ca15e416954af3926f3007d16/helpers/ml.py#L77) Paul Duan used for his winning solution to the [Amazon Employee Access Challenge](https://www.kaggle.com/c/amazon-employee-access-challenge).

```python
>>> from sklearn import datasets
>>> from sklearn import metrics
>>> from sklearn import model_selection
>>> import xam

>>> X, y = datasets.load_digits(n_class=2, return_X_y=True)
>>> X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.5, random_state=42)

>>> model = xam.linear_model.AUCRegressor()
>>> model.fit(X_train, y_train)

>>> train_score = metrics.roc_auc_score(y_train, model.predict(X_train))
>>> test_score = metrics.roc_auc_score(y_test, model.predict(X_test))

>>> print('Train score: {:.2f}'.format(train_score))
Train score: 1.00

>>> print('Test score: {:.2f}'.format(test_score))
Test score: 1.00

```
