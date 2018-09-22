# Preprocessing

## Binning

### Bayesian blocks binning

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

### Equal frequency binning

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

### Equal width binning

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

### Minimum Description Length Principle (MDLP) binning

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


## Groupby transformer

```python
>>> import pandas as pd
>>> from sklearn import preprocessing
>>> import xam

>>> df = pd.DataFrame({
...     'a': [1, None, 3, 3, 3, 3],
...     'b': [4, None, 5, 5, None, 7],
...     'c': [1, 1, 1, 2, 2, 2],
... })

>>> df
     a    b  c
0  1.0  4.0  1
1  NaN  NaN  1
2  3.0  5.0  1
3  3.0  5.0  2
4  3.0  NaN  2
5  3.0  7.0  2

>>> imp = xam.preprocessing.GroupbyTransformer(
...     base_transformer=preprocessing.Imputer(strategy='mean'),
...     by='c'
... )

>>> imp.fit_transform(df)
     a    b  c
0  1.0  4.0  1
1  2.0  4.5  1
2  3.0  5.0  1
3  3.0  5.0  2
4  3.0  6.0  2
5  3.0  7.0  2

```

## One-hot encoding

```python
>>> import pandas as pd
>>> import xam


>>> df = pd.DataFrame({
...     'cat': ['b', 'a', 'a', 'c'],
...     'num': [1, 2, 3, 4]
... })

>>> oh = xam.preprocessing.OneHotEncoder(['cat'])
>>> oh.fit_transform(df)
   num  cat_a  cat_b  cat_c
0    1    0.0    1.0    0.0
1    2    1.0    0.0    0.0
2    3    1.0    0.0    0.0
3    4    0.0    0.0    1.0

```

## Resampling

See this [blog post](https://maxhalford.github.io/subsampling-1/).

```python
>>> import numpy as np
>>> import pandas as pd
>>> import scipy as sp
>>> import xam

>>> np.random.seed(0)

>>> train = pd.DataFrame({
...     'x': np.random.beta(1.5, 2, size=1000),
...     'y': np.random.randint(0, 2, 1000)
... })

>>> test = pd.DataFrame({
...     'x': np.random.beta(2, 1.5, size=1000),
...     'y': np.random.randint(0, 2, 1000)
... })

# Calculate Kullback–Leibler divergence between the train and the test data
>>> sp.stats.entropy(
...     np.histogram(train['x'], bins=30)[0],
...     np.histogram(test['x'], bins=30)[0]
... )  # doctest: +ELLIPSIS
0.252074...

>>> resampler = xam.preprocessing.DistributionResampler(column='x', sample_frac=0.5, seed=0)
>>> resampler.fit(test)

>>> sample = resampler.transform(train)

# The Kullback–Leibler divergence between sample and test is now lower
>>> sp.stats.entropy(
...     np.histogram(sample['x'], bins=30)[0],
...     np.histogram(test['x'], bins=30)[0]
... )  # doctest: +ELLIPSIS
0.073617...

```
