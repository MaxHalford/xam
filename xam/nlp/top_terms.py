import numpy as np
from sklearn import base
from sklearn import utils


class TopTermsClassifier(base.BaseEstimator, base.ClassifierMixin):

    def __init__(self, n_terms=10):
        self.n_terms = n_terms

    def fit(self, X, y=None, **fit_params):

        # scikit-learn checks
        X, y = utils.check_X_y(X, y, accept_sparse='csr', order='C')

        n_terms = min(self.n_terms, X.shape[1])

        # Get a list of unique labels from y
        labels = np.unique(y)

        # Determine the n top terms per class
        self.top_terms_per_class_ = {
            c: set(np.argpartition(np.sum(X[y == c], axis=0), -n_terms)[-n_terms:])
            for c in labels
        }

        # Return the classifier
        return self

    def _predict(self, x):

        # Find the terms in the document
        terms = set(np.where(x > 0)[0])

        # Find the class that has the most top words in common with the document
        return max(
            self.top_terms_per_class_.keys(),
            key=lambda c: len(set.intersection(terms, self.top_terms_per_class_[c]))
        )

    def predict(self, X):

        # scikit-learn checks
        X = utils.check_array(X, accept_sparse='csr', order='C')

        return np.array([self._predict(x) for x in X])
