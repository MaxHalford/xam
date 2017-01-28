from sklearn import datasets, metrics, model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

from xam.stacking import StackingRegressor


boston = datasets.load_boston()
X, y = boston.data, boston.target

m1 = KNeighborsRegressor(n_neighbors=1)
m2 = LinearRegression()
m3 = Ridge(alpha = .5)

sclf = StackingRegressor(models=[m1, m2, m3], meta_model=RandomForestRegressor(random_state=1))

for clf, label in zip(sclf.models + [sclf],
                      ['KNN',
                       'Random Forest',
                       'Ridge regression',
                       'StackingRegressor']):

    scores = model_selection.cross_val_score(clf, X, y, cv=3, scoring='neg_mean_absolute_error')
    print('Accuracy: %0.2f (+/- %0.2f) [%s]' % (-scores.mean(), scores.std(), label))
