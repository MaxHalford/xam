# Data science knowledge/tricks


## Vocabulary

- *IV*: independent variable
- *DV*: dependent variable


## Missing values

- Replace by mean, median or most frequent value
- [Random Forest imputation](http://math.furman.edu/~dcs/courses/math47/R/library/randomForest/html/rfImpute.html)


## Feature engineering

### Binning continuous variables

- [Minimum description length principle](https://arxiv.org/abs/math/0406077)

### String encoding

- Use label encoding for preserving order
- Use one-hot encoding if order does not matter

For one-hot encoding be careful that the dimensionality doesn't blow up.

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


## Feature selection

| IV/DV       | Categorical | Continuous    |
|-------------|-------------|---------------|
| Categorical | Chi Square  | t-test, ANOVA |
| Continuous  | LDA, QDA    | Regression    |

- When the DV is continuous, there is a parametric test for when the DV follows a normal distribution and non-parametric test for when it does not.


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



