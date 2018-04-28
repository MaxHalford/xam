# Data science knowledge/tricks

## Ensemble

(not too sure about the exact vocabulary)

- Blending is averaging predictions
- Bagging is averaging predictions with models trained on different folds with replacement
- Pasting is the same as bagging but without replacement
- Bumping is when a model is trained on different folds and the one that performs the best on the original dataset is kept
- Stacking is training a model on predictions made by other models

## Missing values

- Replace by mean, median or most frequent value
- [Random Forest imputation](http://math.furman.edu/~dcs/courses/math47/R/library/randomForest/html/rfImpute.html)

## Feature engineering

### Temporal features

Day of week, hours, minutes, are cyclic ordinal features; cosine and sine transforms should be used to express the cycle. See [this StackEchange discussion](https://datascience.stackexchange.com/questions/5990/what-is-a-good-way-to-transform-cyclic-ordinal-attributes).

```python
from math import sin, pi

hours = list(range(24))

hours_cos = [cos(pi * h / 24) for h in hours]
hours_sin = [sin(pi * h / 24) for h in hours]
```

### Binning continuous variables

- [Minimum description length principle (entropy)](https://arxiv.org/abs/math/0406077)

### Encoding categorical variables

- One-hot encoding
- [Target encoding](http://delivery.acm.org/10.1145/510000/507538/p27-micci-barreca.pdf?ip=195.220.58.237&id=507538&acc=ACTIVE%20SERVICE&key=7EBF6E77E86B478F%2EDD49F42520D8214D%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&CFID=815531231&CFTOKEN=41271394&__acm__=1507647876_89ea73f9273f9f852423613baaa9f9c8)
- [Feature embedding](https://arxiv.org/pdf/1604.06737.pdf)

### Adstock transformation

Use adstock transformation to take into account lag effects when measure marketing campaign impacts.

```python
advertising = [6, 27, 0, 0, 20, 0, 20] # Marketing campaign intensities

for i in range(1, len(advertising)):
    advertising[i] += advertising[i-1] * 0.5

print(advertising)
```

```sh
>>> [6, 30.0, 15.0, 7.5, 23.75, 11.875, 25.9375]
```


## Dealing with unbalanced classes

- [Read this](https://svds.com/learning-imbalanced-classes/)
- Try under-sampling if there is a lot of data
- Try over-sampling if there is not a lot of data
- Alway under/over-sample on the training set. Don't apply it on the entire set before doing a train/test split, if you do duplicates will exist between the two sets and the scores will be skewed
- Instead of predicting a class predict a probability and use a manual threshold to increase/reduce precision and recall as you wish
- Use weights/costs
- Limit the over-represented class


## Timeseries forecasting

- Use [time series cross-validation](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) (explanatory diagram [here](http://robjhyndman.com/hyndsight/tscv/))


### Spectral analysis for uncovering recurring patterns


## Kaggle tricks

- Adversarial validation can help making relevant cross-validation splits
- Pseudo-labeling by augmenting the training set with part of the labeled test set
