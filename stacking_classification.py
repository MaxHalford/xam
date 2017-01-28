from sklearn import datasets, metrics, model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from xam.stacking import StackingClassifier


iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

m1 = KNeighborsClassifier(n_neighbors=1)
m2 = RandomForestClassifier(random_state=1)
m3 = GaussianNB()

stack = StackingClassifier(models=[m1, m2, m3], meta_model=LogisticRegression())

for clf, label in zip(stack.models + [stack],
                      ['KNN',
                       'Random Forest',
                       'Naïve Bayes',
                       'StackingClassifier']):

    scores = model_selection.cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    print('Accuracy: %0.2f (+/- %0.2f) [%s]' % (scores.mean(), scores.std(), label))
