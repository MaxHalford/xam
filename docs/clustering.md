# Clustering

## Cross-chain algorithm

This is a clustering algorithm I devised at one of my internships for matching customers with multiple accounts. The idea was that some accounts shared some supposedly unique information -- eg. the phone number -- then we would consider those accounts as one single customer. In the following example, the first customer has three accounts; the first account shares the first variable with the second and the second account shares the second variable with the third. The first and third account share no information but they are linked by the second account and form a chain, hence the name of the algorithm.

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
