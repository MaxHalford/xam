import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y


class CrossChainClusterer(BaseEstimator, ClusterMixin):

    def __init__(self, n_terms=10):
        self.n_terms = n_terms

        self.classes_ = None
        self.top_terms_per_class_ = None

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # Check the n_terms attribute is an int and isn't out of bounds
        not_an_int = not isinstance(self.n_terms, int)
        out_of_bounds = self.n_terms < 0

        if not_an_int or out_of_bounds:
            raise ValueError('n_terms should be an int comprised between 0 and the number of terms')

        n_terms = min(self.n_terms, X.shape[1])

        # Determine the n top terms per class
        self.top_terms_per_class_ = {
            c: set(np.argpartition(np.sum(X[y == c], axis=0), -n_terms)[-n_terms:])
            for c in self.classes_
        }

        # Return the classifier
        return self

    def _find_class(self, x):

        # Find the terms in the document
        terms = set(np.where(x > 0)[0])

        # Find the class that has the most top words in common with the document
        return max(
            self.top_terms_per_class_.keys(),
            key=lambda c: len(set.intersection(terms, self.top_terms_per_class_[c]))
        )

    def predict(self, X):
        try:
            getattr(self, 'top_terms_per_class_')
        except AttributeError:
            raise RuntimeError("The classifier has to be fitted before it can be able to predict")

        return [self._find_class(x) for x in X]



clusters = []

def merge(index=0, cluster=None, depth=0):

    new_user = users[index]
    del users[index]

    if cluster is None:
        cluster = [new_user]

    # For each coordinate, find the users that share it
    for key, value in new_user.items():
        for i, user in enumerate(users):
            if user[key] == value:
                cluster.append(user)
                merge(index=i, cluster=cluster, depth=depth+1)

    if depth == 0:
        clusters.append(cluster)

    return

while len(users) > 0:
    merge()

for i, cluster in enumerate(clusters):
    print(i, cluster)
