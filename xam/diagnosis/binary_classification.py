import numpy as np
import pandas as pd

from .base import BaseDiagnosis


class BinaryClassificationDiagnosis(BaseDiagnosis):

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

    def true_positives(self, X, y, sort_by_probability=True):
        return self._filter_predictions(X, y, 1, 1, sort_by_probability)

    def true_negatives(self, X, y, sort_by_probability=True):
        return self._filter_predictions(X, y, 0, 0, sort_by_probability)

    def false_positives(self, X, y, sort_by_probability=True):
        return self._filter_predictions(X, y, 0, 1, sort_by_probability)

    def false_negatives(self, X, y, sort_by_probability=True):
        return self._filter_predictions(X, y, 1, 0, sort_by_probability)
