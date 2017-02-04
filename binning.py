from sklearn import datasets

from xam.binning import MDLPBinner


iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

binner = MDLPBinner()
binner.fit(X, y)
X_discrete = binner.transform(X)

print(X_discrete)
print(y)
