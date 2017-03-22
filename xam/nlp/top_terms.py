import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y


class TopTermsClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_terms=10):
        # Parameters
        self.n_terms = n_terms

        # Attributes
        self.top_terms_per_class_ = None

    def fit(self, X, y):
        # Check that X and y have correct shapes
        X, y = check_X_y(X, y)

        n_terms = min(self.n_terms, X.shape[1])

        # Get a list of unique labels from y
        labels = unique_labels(y)

        # Determine the n top terms per class
        self.top_terms_per_class_ = {
            c: set(np.argpartition(np.sum(X[y == c], axis=0), -n_terms)[-n_terms:])
            for c in labels
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
        return [self._find_class(x) for x in X]
