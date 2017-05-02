# Data science knowledge/tricks


## Vocabulary

- *IV*: independent variable
- *DV*: dependent variable


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

### String encoding

- Use label encoding if order matters (ordinal values)
- Use one-hot encoding if order does not matter (nominal values)

For one-hot encoding be careful that the dimensionality doesn't blow up; also expect the training time to increase because of the added columns.

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

### Temporal cross-validation

### Spectral analysis for uncovering recurring patterns



