from collections import defaultdict

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import Imputer
from sklearn.utils import check_array


def _get_mask(X, value_to_mask):
    """Compute the boolean mask X == missing_values."""
    if value_to_mask == 'NaN' or np.isnan(value_to_mask):
        return np.isnan(X)
    else:
        return X == value_to_mask


class ConditionalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, groupby_col=0, missing_values='NaN', strategy='mean'):
        self.groupby_col = groupby_col
        self.missing_values = missing_values
        self.strategy = strategy

    def fit(self, X, y=None, **fit_params):

        # scikit-learn checks
        X = check_array(X, force_all_finite=False)

        features = np.hstack((X[:, :self.groupby_col], X[:, self.groupby_col+1:]))
        y = np.hstack(X[:, self.groupby_col])

        self.classes_ = np.unique(y)
        # Store one Imputer per column in a dict
        self.imputers_ = defaultdict(dict)

        # Iterate over the columns and fit an Imputer on the rows grouped by the classes in y
        for i, x in enumerate(features.T):
            # Determine the null values in the column
            null_mask = _get_mask(x, self.missing_values)
            for c in self.classes_:
                non_nulls = x[~null_mask & (y == c)]
                if non_nulls.size > 0:
                    self.imputers_[i][c] = Imputer(strategy=self.strategy).fit(
                        non_nulls.reshape(-1, 1)
                    )
        return self

    def transform(self, X, y=None):

        # scikit-learn checks
        X = check_array(X, force_all_finite=False)

        if X.shape[1] != len(self.imputers_) + 1:
            raise ValueError("X has different shape than during fitting. "
                             "Expected %d, got %d." % (len(self.imputers_), X.shape[1]))

        # Iterate over the columns
        for i, col in enumerate(X.T):
            # The grouped by column does not have to be imputed
            if i == self.groupby_col:
                continue
            # Determine the null values in the column
            null_mask = _get_mask(col, self.missing_values)
            # Iterate over the classes in y
            for c in self.classes_:
                # Determine the rows matching the class
                class_fill_mask = null_mask & (X[:, self.groupby_col] == c)
                # If there are missing values for the class then apply the corresponding Imputer
                if np.any(class_fill_mask):
                    X[:, i][class_fill_mask] = self.imputers_[i][c].transform(
                        X[:, i][class_fill_mask].reshape(-1, 1)
                    )[:, 0]
        return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X, y)
