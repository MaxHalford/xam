import numpy as np
import pandas as pd

from .base import ErrorAnalyser


class BinaryClassificationErrorAnalyser(ErrorAnalyser):

    """
    from sklearn import datasets
    from sklearn import model_selection
    from sklearn import svm
    import xam


    X, y = datasets.load_digits(n_class=2, return_X_y=True)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5, random_state=42)

    clf = svm.SVC(probability=True)

    clf.fit(X_train, y_train)

    analyser = xam.error_analysis.BinaryClassificationErrorAnalyser(clf)

    observations, probabilites = analyser.get_false_positives(X_test, y_test)

    print(observations.shape)

    for proba in probabilites:
        print(proba)
    """

    def __init__(self, estimator):
        super().__init__(estimator=estimator)

    def _filter_predictions(self, X, y, true_val, pred_val, sort_by_probability):
        y_pred_proba = self.estimator.predict_proba(X)[:, 1]
        y_pred = y_pred_proba > 0.5
        # Create a mask to filter rows accordingly
        mask = (y == true_val) & (y_pred == pred_val)
        if sort_by_probability:
            # Create a list to order rows by increasing probability
            sorted_idxs = np.argsort(y_pred_proba[mask])
            if isinstance(X, pd.DataFrame):
                return X.reindex(X.index[mask][sorted_idxs]), np.sort(y_pred_proba[mask])
            return X[mask][sorted_idxs], np.sort(y_pred_proba[mask])
        return X[mask], y_pred_proba[mask]

    def get_true_positives(self, X, y, sort_by_probability=True):
        """Return true positives ordered by probability of label being 1"""
        return self._filter_predictions(X, y, 1, 1, sort_by_probability)

    def get_true_negatives(self, X, y, sort_by_probability=True):
        """Return true negatives ordered by probability of label being 0"""
        return self._filter_predictions(X, y, 0, 0, sort_by_probability)

    def get_false_positives(self, X, y, sort_by_probability=True):
        """Return false positives ordered by probability of label being 1"""
        return self._filter_predictions(X, y, 0, 1, sort_by_probability)

    def get_false_negatives(self, X, y, sort_by_probability=True):
        """Return false negatives ordered by probability of label being 0"""
        return self._filter_predictions(X, y, 1, 0, sort_by_probability)
