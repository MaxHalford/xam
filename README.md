# xam [![Build Status](https://travis-ci.org/MaxHalford/xam.svg?branch=master)](https://travis-ci.org/MaxHalford/xam)

xam is my personal data science and machine learning toolbox. It is written in Python 3 and built around mainstream libraries such as pandas and scikit-learn.


## Installation

- [Install Anaconda for Python 3.x](https://www.continuum.io/downloads)
- Run `pip install git+https://github.com/MaxHalford/xam` in a terminal


## Usage examples

The following snippets serve as documentation, examples and tests (through the use of [doctests](https://pymotw.com/2/doctest/)). Again, this is for my personal use so the documentation is not very detailed.

### Preprocessing

**Column selection**

Transformer that extracts one or more columns from a dataframe; is useful for applying a Transformer on a subset of features in a pipeline.

```python
>>> import pandas as pd
>>> from xam import preprocessing

>>> df = pd.DataFrame({'a': [1, 1, 1], 'b': [2, 2, 2], 'c': [3, 3, 3]})

>>> preprocessing.ColumnSelector('a').fit_transform(df)
0    1
1    1
2    1
Name: a, dtype: int64

>>> preprocessing.ColumnSelector(['b', 'c']).fit_transform(df)
   b  c
0  2  3
1  2  3
2  2  3

```

**Column transformer**

Transformer that applies a provided function to each value in a series.

```python
>>> import pandas as pd
>>> from xam import preprocessing

>>> df = pd.DataFrame({'a': [1, 1, 1], 'b': [2, 2, 2]})

>>> preprocessing.ColumnTransformer(lambda x: 2 * x).fit_transform(df)
array([[ 2.,  4.],
       [ 2.,  4.],
       [ 2.,  4.]])

```

**DataFrame transformer**

By design scikit-learn Transformers output numpy nd-arrays, the `DataFrameTransformer` can be used in a pipeline to return pandas dataframes if needed.

```python
```python
>>> import pandas as pd
>>> from sklearn.pipeline import Pipeline
>>> from xam import preprocessing

>>> df = pd.DataFrame({'a': [1, 1, 1], 'b': [2, 2, 2]})

>>> pipeline = Pipeline([
...    ('transform', preprocessing.ColumnTransformer(lambda x: 2 * x)),
...    ('dataframe', preprocessing.DataFrameTransformer(index=df.index, columns=df.columns))
... ])

>>> pipeline.fit_transform(df)
     a    b
0  2.0  4.0
1  2.0  4.0
2  2.0  4.0

```

**Bayesian blocks binning**

Heuristically determines the number of bins to use for continuous variables, see this [blog post](https://jakevdp.github.io/blog/2012/09/12/dynamic-programming-in-python/) for details.

```python
>>> import numpy as np
>>> from scipy import stats
>>> from xam import preprocessing

>>> np.random.seed(0)
>>> x = np.concatenate([
...     stats.cauchy(-5, 1.8).rvs(500),
...     stats.cauchy(-4, 0.8).rvs(2000),
...     stats.cauchy(-1, 0.3).rvs(500),
...     stats.cauchy(2, 0.8).rvs(1000),
...     stats.cauchy(4, 1.5).rvs(500)
... ])
>>> x = x[(x > -15) & (x < 15)].reshape(-1, 1)
>>> binner = preprocessing.BayesianBlocksBinner()
>>> binner.fit_transform(X=x)[:10]
array([[ 6],
       [ 8],
       [ 7],
       [ 6],
       [ 5],
       [ 7],
       [ 5],
       [13],
       [20],
       [ 4]])

```

**Equal frequency binning**

Transformer that bins continuous data into `n_bins` of equal frequency.

```python
>>> import numpy as np
>>> from xam import preprocessing

>>> np.random.seed(42)
>>> mu, sigma = 0, 0.1
>>> x = np.random.normal(mu, sigma, 10).reshape(-1, 1)

>>> binner = preprocessing.EqualFrequencyBinner(n_bins=5)
>>> binner.fit_transform(X=x)
array([[2],
       [1],
       [3],
       [4],
       [0],
       [1],
       [4],
       [3],
       [0],
       [2]])

```

**Equal width binning**

Transformer that bins continuous data into `n_bins` of equal width.

```python
>>> import numpy as np
>>> from xam import preprocessing

>>> np.random.seed(42)
>>> mu, sigma = 0, 0.1
>>> x = np.random.normal(mu, sigma, 10).reshape(-1, 1)

>>> binner = preprocessing.EqualWidthBinner(n_bins=5)
>>> binner.fit_transform(X=x)
array([[2],
       [0],
       [2],
       [4],
       [0],
       [0],
       [5],
       [3],
       [0],
       [2]])

```

**Minimum Description Length Principle (MDLP) binning**

```python
>>> from sklearn import datasets
>>> from xam import preprocessing

>>> iris = datasets.load_iris()
>>> X, y = iris.data[:, 1:3], iris.target

>>> binner = preprocessing.MDLPBinner()
>>> binner.fit_transform(X, y)[:10]
array([[1, 0],
       [0, 0],
       [0, 0],
       [0, 0],
       [1, 0],
       [1, 0],
       [1, 0],
       [1, 0],
       [0, 0],
       [0, 0]])

```


### Clustering

**Cross-chain algorithm**

This is a clustering algorithm I devised at one of my internships for matching customers with multiple accounts. The idea was to that if there accounts shared some information - eg. the phone number - then we would count those accounts as one single customer. In the following example, the first customer has three accounts; the first account shares the first variable with the second and the second account shares the second variable with the third. The first and third account share no information but they are linked by the second account and form a chain, hence the name of the algorithm.

```python
>>> import numpy as np
>>> from xam import clustering

>>> X = np.array([
...     # First expected cluster
...     [0, 1],
...     [0, 2],
...     [1, 2],
...     # Third expected cluster
...     [4, 3],
...     # Second expected cluster
...     [3, 4],
...     [2, 4],
... ])

>>> clustering.CrossChainClusterer().fit_predict(X)
[0, 0, 0, 1, 2, 2]

```


### Model stacking

**Classification**

Model stacking for classification as described in this [Kaggle blog post](http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/).

```python
>>> from sklearn import datasets, metrics, model_selection
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.naive_bayes import GaussianNB
>>> from sklearn.neighbors import KNeighborsClassifier
>>> from xam import stacking

>>> iris = datasets.load_iris()
>>> X, y = iris.data[:, 1:3], iris.target

>>> m1 = KNeighborsClassifier(n_neighbors=1)
>>> m2 = RandomForestClassifier(random_state=1)
>>> m3 = GaussianNB()
>>> stack = stacking.StackingClassifier(models=[m1, m2, m3], meta_model=LogisticRegression())

>>> model_names = ['KNN', 'Random Forest', 'Naïve Bayes', 'StackingClassifier']

>>> for clf, label in zip(stack.models + [stack], model_names):
...     scores = model_selection.cross_val_score(clf, X, y, cv=3, scoring='accuracy')
...     print('Accuracy: %0.2f (+/- %0.2f) [%s]' % (scores.mean(), scores.std(), label))
Accuracy: 0.91 (+/- 0.01) [KNN]
Accuracy: 0.91 (+/- 0.06) [Random Forest]
Accuracy: 0.92 (+/- 0.03) [Naïve Bayes]
Accuracy: 0.95 (+/- 0.03) [StackingClassifier]

```

**Regression**

Model stacking for regression as described in this [Kaggle blog post](http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/).

```python
>>> from sklearn import datasets, metrics, model_selection
>>> from sklearn.ensemble import RandomForestRegressor
>>> from sklearn.linear_model import LinearRegression
>>> from sklearn.linear_model import Ridge
>>> from sklearn.neighbors import KNeighborsRegressor
>>> from xam import stacking

>>> boston = datasets.load_boston()
>>> X, y = boston.data[:, :2], boston.target

>>> m1 = KNeighborsRegressor(n_neighbors=1)
>>> m2 = LinearRegression()
>>> m3 = Ridge(alpha=.5)
>>> stack = stacking.StackingRegressor(models=[m1, m2, m3], meta_model=RandomForestRegressor(random_state=1))

>>> model_names = ['KNN', 'Random Forest', 'Ridge regression', 'StackingRegressor']

>>> for clf, label in zip(stack.models + [stack], model_names):
...     scores = model_selection.cross_val_score(clf, X, y, cv=3, scoring='neg_mean_absolute_error')
...     print('MAE: %0.2f (+/- %0.2f) [%s]' % (-scores.mean(), scores.std(), label))
MAE: 7.45 (+/- 1.22) [KNN]
MAE: 7.72 (+/- 2.09) [Random Forest]
MAE: 7.71 (+/- 2.07) [Ridge regression]
MAE: 6.38 (+/- 0.64) [StackingRegressor]

```


### Natural Language Processing (NLP)

**Top-terms classifier**

```python
>>> from sklearn.datasets import fetch_20newsgroups
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> from xam import nlp

>>> cats = ['alt.atheism', 'comp.windows.x']
>>> newsgroups_train = fetch_20newsgroups(subset='train', categories=cats)
>>> newsgroups_test = fetch_20newsgroups(subset='test', categories=cats)

>>> vectorizer = CountVectorizer(stop_words='english', max_df=0.2)

>>> X_train = vectorizer.fit_transform(newsgroups_train.data)
>>> y_train = newsgroups_train.target

>>> X_test = vectorizer.transform(newsgroups_test.data)
>>> y_test = newsgroups_test.target

>>> clf = nlp.TopTermsClassifier(n_terms=50)
>>> clf.fit(X_train.toarray(), y_train).score(X_test.toarray(), y_test)
0.95238095238095233

```


### Time series analysis (TSA)

**Exponential smoothing forecasting**

```python
>>> import datetime as dt
>>> import numpy as np
>>> import pandas as pd
>>> from xam import tsa

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

>>> tsa.SimpleExponentialSmoothingForecaster(alpha).fit(train).predict(test.index)
1960-01-01    415.452445
1960-02-01    414.407201
1960-03-01    413.466481
1960-04-01    412.619833
1960-05-01    411.857849
1960-06-01    411.172064
1960-07-01    410.554858
1960-08-01    409.999372
1960-09-01    409.499435
1960-10-01    409.049491
1960-11-01    408.644542
1960-12-01    408.280088
dtype: float64

>>> tsa.DoubleExponentialSmoothingForecaster(alpha, beta).fit(train).predict(test.index)
1960-01-01    447.564520
1960-02-01    451.786035
1960-03-01    456.007549
1960-04-01    460.229064
1960-05-01    464.450579
1960-06-01    468.672094
1960-07-01    472.893609
1960-08-01    477.115124
1960-09-01    481.336638
1960-10-01    485.558153
1960-11-01    489.779668
1960-12-01    494.001183
dtype: float64

>>> tsa.TripleExponentialSmoothingForecaster(
...     alpha,
...     beta,
...     gamma,
...     season_length=season_length,
...     multiplicative=True
... ).fit(train).predict(test.index)
1960-01-01    407.899644
1960-02-01    389.067806
1960-03-01    458.415393
1960-04-01    448.046660
1960-05-01    471.033884
1960-06-01    543.653030
1960-07-01    623.363220
1960-08-01    634.072374
1960-09-01    525.714489
1960-10-01    462.219296
1960-11-01    407.274166
1960-12-01    452.141880
dtype: float64

```

**Frequency average forecasting**

```python
>>> import datetime as dt
>>> import pandas as pd
>>> from xam import tsa

>>> df = pd.read_csv('datasets/bike-station.csv')
>>> series = pd.Series(
...     data=df['bikes'].tolist(),
...     index=pd.to_datetime(df['moment'], format='%Y-%m-%d %H:%M:%S')
... )

>>> forecaster = tsa.FrequencyAverageForecaster(lambda d: f'{d.weekday()}-{d.hour}')
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


### Utilities

**Datetime range**

```python
>>> import datetime as dt
>>> from xam import util

>>> since = dt.datetime(2017, 3, 22)
>>> until = dt.datetime(2017, 3, 25)
>>> step = dt.timedelta(days=2)
>>> util.datetime_range(since=since, until=until, step=step)
[datetime.datetime(2017, 3, 22, 0, 0), datetime.datetime(2017, 3, 24, 0, 0)]

```

**Intraclass correlation**

```python
>>> from xam import util

>>> x = [1, 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4]
>>> y = ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']
>>> util.intraclass_correlation(x, y)
0.96031746031746024

```

**Subsequence lengths**

```python
>>> from xam import util

>>> sequence = 'appaaaaapa'
>>> lengths = util.subsequence_lengths(sequence)
>>> print(lengths)
{'a': [1, 5, 1], 'p': [2, 1, 2]}

>>> averages = {k: sum(v) / len(v) for k, v in lengths.items()}
>>> print(averages)
{'a': 2.3333333333333335, 'p': 1.6666666666666667}

```


## License

The MIT License (MIT). Please see the [license file](LICENSE) for more information.
