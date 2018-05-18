import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn import model_selection


class KFoldTargetEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None, n_splits=5, shuffle=True, random_state=None, suffix='_mean'):
        self.columns = columns
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.suffix = suffix

    def fit(self, X, y):
        self.k_fold_ = model_selection.KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )
        return self

    def transform(self, X, y):

        if not isinstance(X, pd.DataFrame):
            raise ValueError('X has to be a pandas.DataFrame')

        if not isinstance(y, pd.Series):
            raise ValueError('y has to be a pandas.Series')

        data = pd.concat((X, y), axis='columns')
        y_col = data.columns[-1]
        means = {col: pd.Series() for col in self.columns}

        for fit_idx, val_idx in self.k_fold_.split(data):

            fit, val = data.iloc[fit_idx], data.iloc[val_idx]

            for col in self.columns:

                col_means = fit.groupby(col)[y_col].mean()

                means[col] = pd.concat(
                    (
                        means[col],
                        val[[col]].join(col_means, on=col)[y_col].fillna(col_means.mean())
                    ),
                    axis='rows'
                )

        for col in means:
            X[col + self.suffix] = means[col]

        return X

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X, y)
