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

## Combining features

```python
>>> import pandas as pd
>>> import xam

>>> df = pd.DataFrame({
...     'col_a': ['a', 'b', 'c'],
...     'col_b': ['d', 'e', 'f'],
...     'col_c': ['g', 'h', 'i'],
... })

>>> xam.preprocessing.FeatureCombiner(separator='+', orders=[2, 3]).fit_transform(df)
  col_a col_b col_c col_a+col_b col_a+col_c col_b+col_c col_a+col_b+col_c
0     a     d     g         a+d         a+g         d+g             a+d+g
1     b     e     h         b+e         b+h         e+h             b+e+h
2     c     f     i         c+f         c+i         f+i             c+f+i

```

## Cyclic features

Day of week, hours, minutes, are cyclic ordinal features; cosine and sine transforms should be used to express the cycle. See [this StackEchange discussion](https://datascience.stackexchange.com/questions/5990/what-is-a-good-way-to-transform-cyclic-ordinal-attributes). This transformer returns an array with twice as many columns as the input array; the first columns are the cosine transforms and the last columns are the sine transforms.

```python
>>> import numpy as np
>>> import xam

>>> times = np.array([
...    np.linspace(0, 23, 4),
...    np.linspace(0, 59, 4),
... ]).T

>>> trans = xam.preprocessing.CycleTransformer()
>>> trans.fit_transform(times)
array([[ 1.        ,  1.        ,  0.        ,  0.        ],
       [-0.42261826, -0.46947156,  0.90630779,  0.88294759],
       [-0.64278761, -0.5591929 , -0.76604444, -0.82903757],
       [ 0.96592583,  0.9945219 , -0.25881905, -0.10452846]])

```

## Imputation

### Conditional imputation

Scikit-learn's [`Imputer`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html) transformer is practical for it is an unsupervised method. `ConditionalImputer` makes it possible to apply an `Imputer` in a supervised way. In other words the `Imputer` is applied conditionally on the value of `y`.

```python
>>> import numpy as np
>>> from sklearn.preprocessing import Imputer
>>> import xam

>>> X = np.array([
...     [1,      4,      1],
...     [np.nan, np.nan, 1],
...     [3,      5,      1],
...     [3,      5,      2],
...     [3,      np.nan, 2],
...     [3,      7,      2],
... ])

>>> imp = xam.preprocessing.ConditionalImputer(groupby_col=2, strategy='mean')
>>> imp.fit_transform(X)
array([[ 1. ,  4. ,  1. ],
       [ 2. ,  4.5,  1. ],
       [ 3. ,  5. ,  1. ],
       [ 3. ,  5. ,  2. ],
       [ 3. ,  6. ,  2. ],
       [ 3. ,  7. ,  2. ]])

```

## Likelihood encoding

Based on [this paper](http://delivery.acm.org/10.1145/510000/507538/p27-micci-barreca.pdf?ip=195.220.58.237&id=507538&acc=ACTIVE%20SERVICE&key=7EBF6E77E86B478F%2EDD49F42520D8214D%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&CFID=815531231&CFTOKEN=41271394&__acm__=1507647876_89ea73f9273f9f852423613baaa9f9c8).


```python
>>> import pandas as pd
>>> import xam

>>> X = pd.DataFrame({'x_0': ['a'] * 5 + ['b'] * 5, 'x_1': ['a'] * 9 + ['b'] * 1})
>>> y = pd.Series([1, 1, 1, 1, 0, 1, 0, 0, 0, 0])

>>> be = xam.preprocessing.LikelihoodEncoder(columns=['x_0', 'x_1'], min_samples=3, smoothing=2)
>>> be.fit_transform(X, y)
        x_0       x_1
0  0.719318  0.542382
1  0.719318  0.542382
2  0.719318  0.542382
3  0.719318  0.542382
4  0.719318  0.542382
5  0.280682  0.542382
6  0.280682  0.542382
7  0.280682  0.542382
8  0.280682  0.542382
9  0.280682  0.203072

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
... )
0.25207468085005064

>>> resampler = xam.preprocessing.DistributionResampler(column='x', sample_frac=0.5, seed=0)
>>> resampler.fit(test)

>>> sample = resampler.transform(train)

# The Kullback–Leibler divergence between sample and test is now lower
>>> sp.stats.entropy(
...     np.histogram(sample['x'], bins=30)[0],
...     np.histogram(test['x'], bins=30)[0]
... )
0.073617242561277552

```
