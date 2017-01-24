from sklearn import datasets, metrics, model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from hedgehog.stacking import StackingClassifier


iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)

clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()

sclf = StackingClassifier(models=[clf1, clf2, clf3], meta_model=LogisticRegression())

for clf, label in zip(sclf.models + [sclf],
                      ['KNN',
                       'Random Forest',
                       'Naïve Bayes',
                       'StackingClassifier']):

    scores = model_selection.cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    print('Accuracy: %0.2f (+/- %0.2f) [%s]' % (scores.mean(), scores.std(), label))
