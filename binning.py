from sklearn import datasets

from xam.binning import MDLPBinner


iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

binner = MDLPBinner()
X_discrete = binner.fit_transform(X, y)

print(X_discrete)
print(y)
