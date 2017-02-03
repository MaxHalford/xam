from sklearn import datasets

from xam.discretization import MDLP


iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

discretizer = MDLP()
X_discrete = discretizer.fit_transform(X, y)

print(X_discrete)
print(y)
