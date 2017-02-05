import numpy as np

from xam.clustering import CrossChainClusterer


X = np.array([
    # First expected cluster
    [0, 1],
    [0, 2],
    [1, 2],
    # Third expected cluster
    [4, 3],
    # Second expected cluster
    [3, 4],
    [2, 4],
])

clusterer = CrossChainClusterer().fit(X)

print(clusterer.labels_)

print(clusterer.predict(X))
