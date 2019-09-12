import math
import operator


class ForwardBackwardSelector:
    """Forward-backward feature selector.

    Parameters:
        score_func (callable): Function taking two arrays X and y, and returning score. The
            TokenSelector will attempt to minimize this score.
        n_forward (int): Number of steps going forward.
        n_backward (int): Number of steps going backward.
        verbose (bool): Whether or not to display the progress of the procedure.

    References:
        1. http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.24.4369&rep=rep1&type=pdf

    """

    def __init__(self, score_func, n_forward=1, n_backward=0, verbose=True):

        if n_forward <= n_backward:
            raise ValueError('n_forward must be stricly superior to n_backward')

        if n_forward < 0 or n_backward < 0:
            raise ValueError('n_forward and n_backward must both positive')

        self.n_forward = n_forward
        self.n_backward = n_backward
        self.score_func = score_func
        self.verbose = verbose

    def _print(self, msg):
        if self.verbose:
            print(msg)

    def fit(self, X, y):

        self.best_combo_ = set()
        self.best_score_ = math.inf
        features = set(X.columns)

        while features:

            # Go forward
            for _ in range(self.n_forward):

                # Measure the score when adding each feature to the current combination
                scores = {
                    feature: self.score_func(X[list(self.best_combo_ | {feature})], y)
                    for feature in features
                }

                # Find the feature which brings the lowest score (or the least high!)
                best_feature, score = min(scores.items(), key=operator.itemgetter(1))

                # Stop if no feature brought any gain
                if score >= self.best_score_:
                    self._print('No further improvement was found, stopping')
                    return self

                # Update the current best score and the best combination of features
                self.best_score_ = score
                self.best_combo_.add(best_feature)

                # Remove the best found feature from the pool of features
                features.remove(best_feature)

                self._print(f'Added feature "{best_feature}", new best score is {self.best_score_:.5f}')

            for _ in range(self.n_backward):

                # Measure the score when adding each feature to the current combination
                scores = {
                    feature: self.score_func(X[list(self.best_combo_ - {feature})], y)
                    for feature in self.best_combo_
                }

                # Find the feature which brings the lowest score (or the least high!)
                best_feature, score = min(scores.items(), key=operator.itemgetter(1))

                # Stop if no feature brought any gain
                if score >= self.best_score_:
                    break

                # Update the current best score and the best combination of features
                self.best_score_ = score
                self.best_combo_.remove(best_feature)

                # Add the best found feature to the pool of features
                features.add(best_feature)

                self._print(f'Removed feature "{best_feature}", new best score is {self.best_score_:.5f}')

    def transform(self, X):
        return X[list(sorted(self.best_combo_))]
