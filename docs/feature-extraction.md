# Feature extraction

## Combining features

```python
>>> import pandas as pd
>>> import xam

>>> df = pd.DataFrame({
...     'col_a': ['a', 'b', 'c'],
...     'col_b': ['d', 'e', 'f'],
...     'col_c': ['g', 'h', 'i'],
... })

>>> xam.feature_extraction.FeatureCombiner(separator='+', orders=[2, 3]).fit_transform(df)  # doctest:+ELLIPSIS
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

>>> trans = xam.feature_extraction.CycleTransformer()
>>> trans.fit_transform(times)
array([[ 1.        ,  1.        ,  0.        ,  0.        ],
       [-0.42261826, -0.46947156,  0.90630779,  0.88294759],
       [-0.64278761, -0.5591929 , -0.76604444, -0.82903757],
       [ 0.96592583,  0.9945219 , -0.25881905, -0.10452846]])

```


## K-fold target encoding

```python
>>> import pandas as pd
>>> import xam

>>> X = pd.DataFrame({'x_0': ['a'] * 5 + ['b'] * 5, 'x_1': ['a'] * 9 + ['b'] * 1})
>>> y = pd.Series([1, 1, 1, 1, 0, 1, 0, 0, 0, 0])

>>> encoder = xam.feature_extraction.KFoldTargetEncoder(
...     columns=['x_0', 'x_1'],
...     suffix='',
...     random_state=42
... )
>>> be.fit(X, y).transform(X)
    x_0       x_1
0  0.75  0.428571
1  0.75  0.571429
2  0.75  0.571429
3  0.75  0.571429
4  1.00  0.625000
5  0.00  0.428571
6  0.25  0.571429
7  0.25  0.571429
8  0.25  0.571429
9  0.25  0.625000

```


## Smooth target encoding

Based on [this](https://www.wikiwand.com/en/Additive_smoothing).

```python
>>> import pandas as pd
>>> import xam

>>> X = pd.DataFrame({'x_0': ['a'] * 5 + ['b'] * 5, 'x_1': ['a'] * 9 + ['b'] * 1})
>>> y = pd.Series([1, 1, 1, 1, 0, 1, 0, 0, 0, 0])

>>> encoder = xam.feature_extraction.SmoothTargetEncoder(
...     columns=['x_0', 'x_1'],
...     prior_weight=3,
...     suffix=''
... )
>>> encoder.fit(X, y).transform(X)
      x_0       x_1
0  0.6875  0.486111
1  0.6875  0.486111
2  0.6875  0.486111
3  0.6875  0.486111
4  0.6875  0.486111
5  0.3125  0.486111
6  0.3125  0.486111
7  0.3125  0.486111
8  0.3125  0.486111
9  0.3125  0.208333

```
