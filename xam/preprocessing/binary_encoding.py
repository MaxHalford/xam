import math

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array


class BinaryEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, **fit_params):

        # scikit-learn checks
        X = check_array(X)

        self.n_cols_ = dict()
        self.codes_ = dict()

        # Iterate over the columns
        for i, x in enumerate(X.T):

            # Determine the number of digits that are needed to encode all the labels in a column
            labels = np.unique(x)
            self.n_cols_[i] = math.ceil(math.log2(len(labels)+1))

            # Determine the binary representation for each label in the column
            self.codes_[i] = {
                label: np.array([
                    int(b)
                    for b in list('{:b}'.format(j+1).zfill(self.n_cols_[i]))
                ])
                for j, label in enumerate(sorted(labels))
            }

        return self

    def transform(self, X, y=None):

        # scikit-learn checks
        X = check_array(X)

        if X.shape[1] != len(self.n_cols_):
            raise ValueError("X has different shape than during fitting. "
                             "Expected %d, got %d." % (len(self.n_cols_), X.shape[1]))

        return np.hstack([
            np.vstack([
                # Use a default value if the label has never been seen
                self.codes_[i].get(v, np.zeros(shape=(self.n_cols_[i])))
                for v in x
            ])
            for i, x in enumerate(X.T)
        ])
