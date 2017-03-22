from sklearn import datasets
from sklearn.utils.estimator_checks import check_estimator

from xam.binning import MDLPBinner, EqualWidthBinner


iris = datasets.load_iris()
X, y = iris.data, iris.target

print(check_estimator(MDLPBinner))
