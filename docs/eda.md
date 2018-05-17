# Exploratory data analysis

## Feature importance

The `feature_importance` method returns two dataframes that contain feature importance metrics that depend on the types of the feature/target

| Feature/Task         | Classification         | Regression          |
|----------------------|------------------------|---------------------|
| Categorical          | Chi²-test + Cramér's V | F-test              |
| Numerical            | F-test                 | Pearson correlation |

Additionally [mutual information](https://www.wikiwand.com/en/Mutual_information) can be used in each case.

- [Comparison of F-test and mutual information](http://scikit-learn.org/stable/auto_examples/feature_selection/plot_f_test_vs_mi.html)

Classification.

```python
>>> import pandas as pd
>>> from sklearn import datasets
>>> import xam

>>> iris = datasets.load_iris()
>>> features = pd.DataFrame(iris.data, columns=iris.feature_names)
>>> features['sepal length (cm)'] = features['sepal length (cm)'] > 5.5
>>> target = pd.Series(iris.target)

>>> cont_imp, disc_imp = xam.eda.feature_importance_classification(features, target, random_state=1)

>>> cont_imp.sort_values('f_p_value')  # doctest: +SKIP
                   f_statistic     f_p_value  mutual_information
petal length (cm)  1179.034328  3.051976e-91            0.990061
petal width (cm)    959.324406  4.376957e-85            0.977279
sepal width (cm)     47.364461  1.327917e-16            0.256295

>>> disc_imp.sort_values('chi2_p_value')  # doctest: +SKIP
                   chi2_statistic  chi2_p_value  cramers_v  mutual_information
sepal length (cm)        98.11883  4.940452e-22   0.803139            0.386244

```

Regression.

```python
>>> import pandas as pd
>>> from sklearn import datasets
>>> import xam

>>> boston = datasets.load_boston()
>>> features = pd.DataFrame(boston.data, columns=boston.feature_names)
>>> features['CHAS'] = features['CHAS'].astype(int)
>>> target = pd.Series(boston.target)

>>> cont_imp, disc_imp = xam.eda.feature_importance_regression(features, target, random_state=1)

>>> cont_imp.sort_values('pearson_r_p_value')
         pearson_r  pearson_r_p_value  mutual_information
LSTAT    -0.737663       5.081103e-88            0.666882
RM        0.695360       2.487229e-74            0.526456
PTRATIO  -0.507787       1.609509e-34            0.453291
INDUS    -0.483725       4.900260e-31            0.471507
TAX      -0.468536       5.637734e-29            0.363694
NOX      -0.427321       7.065042e-24            0.456947
CRIM     -0.385832       2.083550e-19            0.334339
RAD      -0.381626       5.465933e-19            0.217623
AGE      -0.376955       1.569982e-18            0.311285
ZN        0.360445       5.713584e-17            0.195153
B         0.333461       1.318113e-14            0.161861
DIS       0.249929       1.206612e-08            0.295207

>>> disc_imp.sort_values('mutual_information')
      f_statistic  f_p_value  mutual_information
CHAS    15.971512   0.000074            0.030825

```
