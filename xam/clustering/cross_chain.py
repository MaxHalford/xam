import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y

from ..base import Model


class CrossChainClusterer(BaseEstimator, ClusterMixin, Model):

    def __init__(self):
        # Attributes
        self.labels_ = None
        self.cluster_coordinates_ = None

    def fit(self, X, y=None):

        (n, p) = X.shape
        label = 0
        self.labels_ = [0] * n
        visited = set()
        self.cluster_coordinates_ = []

        def cross_chain(index, label):
            point = X[index]

            visited.add(index)
            self.labels_[index] = label

            for i in range(p):
                self.cluster_coordinates_[label][i].add(point[i])

            # For each coordinate, find the points that share it
            for i, coord in enumerate(point):
                for j, point in enumerate(X):
                    if j not in visited and point[i] == coord:
                        cross_chain(index=j, label=label)

            return

        while len(visited) < n:
            self.cluster_coordinates_.append([set() for _ in range(p)])
            not_visited = set(range(n)) - visited
            cross_chain(index=min(not_visited), label=label)
            label += 1

        return self

    def predict(self, X, y=None):
        """Predict the label of each element in X.

        Returns -1 for the elements that cannot be matched to any existing cluster.
        """
        labels = [-1] * len(X)

        for i, x in enumerate(X):
            # Go through each cluster
            for j, coord_sets in enumerate(self.cluster_coordinates_):
                found = False
                # Go through each set of coordinates of the cluster
                for k, coord_set in enumerate(coord_sets):
                    if x[k] in coord_set:
                        found = True
                        break
                if found:
                    labels[i] = j
                    break

        return labels

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_

    def check_params(self):
        return

    @property
    def is_fitted(self):
        return all((self.labels_ is not None, self.cluster_coordinates_ is not None))
