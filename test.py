from sklearn import model_selection
from sklearn.linear_model import Lasso
from sklearn.datasets import load_diabetes

from xam.splitting import SplittingEstimator


X, y = load_diabetes(return_X_y=True)


def split(row):
    return row[2] > 0


split_estimator = SplittingEstimator(Lasso(alpha=0.01), split)

scores = model_selection.cross_val_score(Lasso(alpha=0.01), X, y, scoring='r2')

print('{:.3f} (+/- {:.3f})'.format(-scores.mean(), scores.std()))

scores = model_selection.cross_val_score(split_estimator, X, y, scoring='r2')

print('{:.3f} (+/- {:.3f})'.format(-scores.mean(), scores.std()))
