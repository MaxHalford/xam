import lightgbm as lgbm
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import utils


class LGBMCV():

    def __init__(self, cv=model_selection.KFold(n_splits=5, shuffle=True), **kwargs):
        self.cv = cv
        self.lgbm_params = kwargs

    def fit(self, X, y=None, **kwargs):

        self.models_ = []
        feature_names = X.columns if isinstance(X, pd.DataFrame) else list(range(X.shape[1]))
        self.feature_importances_ = pd.DataFrame(index=feature_names)
        self.evals_results_ = {}

        for i, (fit_idx, val_idx) in enumerate(self.cv.split(X, y)):

            # Split the dataset according to the fold indexes
            if isinstance(X, pd.DataFrame):
                X_fit = X.iloc[fit_idx]
                X_val = X.iloc[val_idx]
            else:
                X_fit = X[fit_idx]
                X_val = X[val_idx]

            if isinstance(y, pd.Series):
                y_fit = y.iloc[fit_idx]
                y_val = y.iloc[val_idx]
            else:
                y_fit = y[fit_idx]
                y_val = y[val_idx]

            # https://lightgbm.readthedocs.io/en/latest/Python-API.html#lightgbm.Dataset
            fit_set = lgbm.Dataset(X_fit, y_fit)
            val_set = lgbm.Dataset(X_val, y_val)

            # https://lightgbm.readthedocs.io/en/latest/Python-API.html#lightgbm.train
            self.evals_results_[i] = {}
            model = lgbm.train(
                params=self.lgbm_params,
                train_set=fit_set,
                valid_sets=(fit_set, val_set),
                valid_names=('fit', 'val'),
                evals_result=self.evals_results_[i],
                **kwargs
            )

            # Store the feature importances
            self.feature_importances_['gain_{}'.format(i)] = model.feature_importance('gain')
            self.feature_importances_['split_{}'.format(i)] = model.feature_importance('split')

            # Store the model
            self.models_.append(model)

        return self

    def predict(self, X):

        utils.validation.check_is_fitted(self, ['models_'])

        y = np.zeros(len(X))

        for model in self.models_:
            y += model.predict(X)

        return y / len(self.models_)
