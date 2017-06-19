import numpy as np
import pandas as pd

from .binning.equal_frequency import EqualFrequencyBinner


class DistributionSubsampler():

    def __init__(self, feature=0, sample_frac=0.5, n_bins=100, seed=None):
        super().__init__()
        self.feature = feature
        self.sample_frac = sample_frac
        self.binner = EqualFrequencyBinner(n_bins=n_bins)
        self.seed = None

    def fit(self, X, y=None, **fit_params):

        if isinstance(X, pd.DataFrame):
            self.binner.fit(X[[self.feature]])
        else:
            self.binner.fit(X[:, self.feature].reshape(-1, 1))

    def transform(self, X, y=None):

        if isinstance(X, pd.DataFrame):
            bins = self.binner.transform(X[[self.feature]])[:, 0]
        else:
            bins = self.binner.transform(X[:, self.feature].reshape(-1, 1))[:, 0]

        # Match each observation in the training set to a bin
        bin_counts = np.bincount(bins)

        # Count the number of values in each training set bin
        weights = 1 / np.array([bin_counts[x] for x in bins])

        # Weight each observation in the training set based on which bin it is in
        weights_norm = weights / np.sum(weights)


        if isinstance(X, pd.DataFrame):
            return X.sample(
                frac=self.sample_frac,
                weights=weights_norm,
                replace=False,
                random_state=self.seed
            )
        else:
            weights_norm = weights / np.sum(weights)
            return pd.DataFrame(X).sample(
                frac=self.sample_frac,
                weights=weights_norm,
                replace=False,
                random_state=self.seed
            ).values


#ewb = xam.preprocessing.EqualFrequencyBinner(n_bins=300)
#ewb.fit(test.reshape(-1, 1))
#train_bins = ewb.transform(train.reshape(-1, 1))[:, 0]
#train_bin_counts = Counter(train_bins)
#weights = np.array([1 / train_bin_counts[x] for x in train_bins])
#weights_norm = weights / np.sum(weights)

# Sample from the training set
#np.random.seed(0)
#sample = np.random.choice(train, size=30000, p=weights_norm, replace=False)

#with plt.xkcd():
#    fig, ax = plt.subplots(figsize=(14, 8))
#    sns.kdeplot(train, ax=ax, label='Train');
#    sns.kdeplot(test, ax=ax, label='Test');
#    sns.kdeplot(sample, ax=ax, label='Train resampled');
#    ax.legend();
