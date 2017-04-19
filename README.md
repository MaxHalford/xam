# xam [![Build Status](https://travis-ci.org/MaxHalford/xam.svg?branch=master)](https://travis-ci.org/MaxHalford/xam)

xam is my personal data science and machine learning toolbox. It is written in Python 3 and built around mainstream libraries such as pandas and scikit-learn.


## Installation

- [Install Anaconda for Python 3.x](https://www.continuum.io/downloads)
- Run `pip install git+https://github.com/MaxHalford/xam` in a terminal


## Other Python data science and machine learning toolkits

- [ogrisel/oglearn](https://github.com/ogrisel/oglearn)
- [rasbt/mlxtend](https://github.com/rasbt/mlxtend)


## Usage examples

The following snippets serve as documentation, examples and tests - through the use of [doctests](https://pymotw.com/2/doctest/). Again, this is for my personal use so the documentation is not very detailed.

### Preprocessing

**Column selection**

Transformer that extracts one or more columns from a dataframe; is useful for applying a Transformer on a subset of features in a pipeline.

```python
>>> import pandas as pd
>>> import xam

>>> df = pd.DataFrame({'a': [1, 1, 1], 'b': [2, 2, 2], 'c': [3, 3, 3]})

>>> xam.preprocessing.ColumnSelector('a').fit_transform(df)
0    1
1    1
2    1
Name: a, dtype: int64

>>> xam.preprocessing.ColumnSelector(['b', 'c']).fit_transform(df)
   b  c
0  2  3
1  2  3
2  2  3

```


**Series transformer**

Applies a function to each value in series.

```python
>>> import pandas as pd
>>> from sklearn.pipeline import Pipeline
>>> from xam.preprocessing import ColumnSelector
>>> from xam.preprocessing import SeriesTransformer

>>> df = pd.DataFrame({'a': [1, 1, 1], 'b': [2, 2, 2]})

>>> pipeline = Pipeline([
...    ('extract', ColumnSelector('a')),
...    ('transform', SeriesTransformer(lambda x: 2 * x))
... ])

>>> pipeline.fit_transform(df)
0    2
1    2
2    2
Name: a, dtype: int64

```


**Convert to DataFrame transformer**

By design scikit-learn Transformers output numpy nd-arrays, the `ToDataFrameTransformer` can be used in a pipeline to return pandas dataframes if needed.

```python
>>> import pandas as pd
>>> from sklearn.pipeline import Pipeline
>>> from xam.preprocessing import ColumnSelector
>>> from xam.preprocessing import SeriesTransformer
>>> from xam.preprocessing import ToDataFrameTransformer

>>> df = pd.DataFrame({'a': [1, 1, 1], 'b': [2, 2, 2]})

>>> pipeline = Pipeline([
...    ('extract', ColumnSelector('a')),
...    ('transform', SeriesTransformer(lambda x: 2 * x)),
...    ('dataframe', ToDataFrameTransformer())
... ])

>>> pipeline.fit_transform(df)
   a
0  2
1  2
2  2

```

**Label vectorizer**

One-hot encoder that works in a pipeline.

```python
>>> import pandas as pd
>>> import xam

>>> df = pd.DataFrame({'one': ['a', 'b', 'c'], 'two': ['c', 'a', 'c']})

>>> xam.preprocessing.LabelVectorizer().fit_transform(df)
   one_a  one_b  one_c  two_a  two_c
0      1      0      0      0      1
1      0      1      0      1      0
2      0      0      1      0      1

```

**Lambda transformer**

Will apply a function to the input; this transformer can potentially do anything but you have to properly keep track of your inputs and outputs.

```python
>>> import pandas as pd
>>> import xam

>>> df = pd.DataFrame({'one': ['a', 'a', 'a'], 'two': ['c', 'a', 'c']})

>>> def has_one_c(dataframe):
...    return (dataframe['one'] == 'c') | (dataframe['two'] == 'c')

>>> xam.preprocessing.LambdaTransfomer(has_one_c).fit_transform(df)
0     True
1    False
2     True
dtype: bool

```


**Bayesian blocks binning**

Heuristically determines the number of bins to use for continuous variables, see this [blog post](https://jakevdp.github.io/blog/2012/09/12/dynamic-programming-in-python/) for details.

```python
>>> import numpy as np
>>> from scipy import stats
>>> import xam

>>> np.random.seed(0)
>>> x = np.concatenate([
...     stats.cauchy(-5, 1.8).rvs(500),
...     stats.cauchy(-4, 0.8).rvs(2000),
...     stats.cauchy(-1, 0.3).rvs(500),
...     stats.cauchy(2, 0.8).rvs(1000),
...     stats.cauchy(4, 1.5).rvs(500)
... ])
>>> x = x[(x > -15) & (x < 15)].reshape(-1, 1)
>>> binner = xam.preprocessing.BayesianBlocksBinner()
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
>>> import xam

>>> np.random.seed(42)
>>> mu, sigma = 0, 0.1
>>> x = np.random.normal(mu, sigma, 10).reshape(-1, 1)

>>> binner = xam.preprocessing.EqualFrequencyBinner(n_bins=5)
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
>>> import xam

>>> np.random.seed(42)
>>> mu, sigma = 0, 0.1
>>> x = np.random.normal(mu, sigma, 10).reshape(-1, 1)

>>> binner = xam.preprocessing.EqualWidthBinner(n_bins=5)
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
>>> import xam

>>> iris = datasets.load_iris()
>>> X, y = iris.data[:, 1:3], iris.target

>>> binner = xam.preprocessing.MDLPBinner()
>>> binner.fit_transform(X, y)[:10]
array([[2, 0],
       [1, 0],
       [1, 0],
       [1, 0],
       [2, 0],
       [2, 0],
       [2, 0],
       [2, 0],
       [0, 0],
       [1, 0]])

```


### Clustering

**Cross-chain algorithm**

This is a clustering algorithm I devised at one of my internships for matching customers with multiple accounts. The idea was to that if there accounts shared some information - eg. the phone number - then we would count those accounts as one single customer. In the following example, the first customer has three accounts; the first account shares the first variable with the second and the second account shares the second variable with the third. The first and third account share no information but they are linked by the second account and form a chain, hence the name of the algorithm.

```python
>>> import numpy as np
>>> import xam

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

>>> xam.clustering.CrossChainClusterer().fit_predict(X)
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
>>> import xam

>>> iris = datasets.load_iris()
>>> X, y = iris.data[:, 1:3], iris.target

>>> m1 = KNeighborsClassifier(n_neighbors=1)
>>> m2 = RandomForestClassifier(random_state=1)
>>> m3 = GaussianNB()
>>> stack = xam.stacking.StackingClassifier(
...     models=[m1, m2, m3],
...     meta_model=LogisticRegression()
... )

>>> model_names = ['KNN', 'Random Forest', 'Naïve Bayes', 'StackingClassifier']

>>> for clf, label in zip(stack.models + [stack], model_names):
...     scores = model_selection.cross_val_score(clf, X, y, cv=3, scoring='accuracy')
...     print('Accuracy: %0.2f (+/- %0.2f) [%s]' % (scores.mean(), 1.96 * scores.std(), label))
Accuracy: 0.91 (+/- 0.02) [KNN]
Accuracy: 0.91 (+/- 0.13) [Random Forest]
Accuracy: 0.92 (+/- 0.05) [Naïve Bayes]
Accuracy: 0.95 (+/- 0.06) [StackingClassifier]

```

**Regression**

Model stacking for regression as described in this [Kaggle blog post](http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/).

```python
>>> from sklearn import datasets, metrics, model_selection
>>> from sklearn.ensemble import RandomForestRegressor
>>> from sklearn.linear_model import LinearRegression
>>> from sklearn.linear_model import Ridge
>>> from sklearn.neighbors import KNeighborsRegressor
>>> import xam

>>> boston = datasets.load_boston()
>>> X, y = boston.data[:, :2], boston.target

>>> m1 = KNeighborsRegressor(n_neighbors=1)
>>> m2 = LinearRegression()
>>> m3 = Ridge(alpha=.5)
>>> stack = xam.stacking.StackingRegressor(
...     models=[m1, m2, m3],
...     meta_model=RandomForestRegressor(random_state=1)
... )

>>> model_names = ['KNN', 'Random Forest', 'Ridge regression', 'StackingRegressor']

>>> for clf, label in zip(stack.models + [stack], model_names):
...     scores = model_selection.cross_val_score(clf, X, y, cv=3, scoring='neg_mean_absolute_error')
...     print('MAE: %0.2f (+/- %0.2f) [%s]' % (-scores.mean(), 1.96 * scores.std(), label))
MAE: 7.45 (+/- 2.39) [KNN]
MAE: 7.72 (+/- 4.10) [Random Forest]
MAE: 7.71 (+/- 4.06) [Ridge regression]
MAE: 6.38 (+/- 1.25) [StackingRegressor]

```


### Natural Language Processing (NLP)

**Top-terms classifier**

```python
>>> from sklearn.datasets import fetch_20newsgroups
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> import xam

>>> cats = ['alt.atheism', 'comp.windows.x']
>>> newsgroups_train = fetch_20newsgroups(subset='train', categories=cats)
>>> newsgroups_test = fetch_20newsgroups(subset='test', categories=cats)

>>> vectorizer = CountVectorizer(stop_words='english', max_df=0.2)

>>> X_train = vectorizer.fit_transform(newsgroups_train.data)
>>> y_train = newsgroups_train.target

>>> X_test = vectorizer.transform(newsgroups_test.data)
>>> y_test = newsgroups_test.target

>>> clf = xam.nlp.TopTermsClassifier(n_terms=50)
>>> clf.fit(X_train.toarray(), y_train).score(X_test.toarray(), y_test)
0.95238095238095233

```


### Time series analysis (TSA)

**Exponential smoothing forecasting**

```python
>>> import datetime as dt
>>> import numpy as np
>>> import pandas as pd
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

>>> xam.tsa.SimpleExponentialSmoothingForecaster(alpha).fit(train).predict(test.index)
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

>>> xam.tsa.DoubleExponentialSmoothingForecaster(alpha, beta).fit(train).predict(test.index)
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

>>> xam.tsa.TripleExponentialSmoothingForecaster(
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


### Utilities

**Datetime range**

```python
>>> import datetime as dt
>>> import xam

>>> since = dt.datetime(2017, 3, 22)
>>> until = dt.datetime(2017, 3, 25)
>>> step = dt.timedelta(days=2)
>>> xam.util.datetime_range(since=since, until=until, step=step)
[datetime.datetime(2017, 3, 22, 0, 0), datetime.datetime(2017, 3, 24, 0, 0)]

```

**Intraclass correlation**

```python
>>> import xam

>>> x = [1, 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4]
>>> y = ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']
>>> xam.util.intraclass_correlation(x, y)
0.96031746031746024

```

**Subsequence lengths**

```python
>>> import xam

>>> sequence = 'appaaaaapa'
>>> lengths = xam.util.subsequence_lengths(sequence)
>>> print(lengths)
{'a': [1, 5, 1], 'p': [2, 1, 2]}

>>> averages = {k: sum(v) / len(v) for k, v in lengths.items()}
>>> print(averages)
{'a': 2.3333333333333335, 'p': 1.6666666666666667}

```


## License

The MIT License (MIT). Please see the [license file](LICENSE) for more information.
