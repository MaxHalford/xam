# Data science tricks

## Feature selection

### Continuous -> Continuous

- Pearson's correlation ratio
- [Coefficient of determination](https://www.wikiwand.com/en/Coefficient_of_determination)

### Continuous <-> Nominal

- Intraclass correlation
- [Cohen's D](https://www.wikiwand.com/en/Effect_size#/Cohen.27s_d)
- ANOVA f-test (t-test for two classes)

### Nominal -> Nominal

- Chi-square test


## Marketing Mix Modeling

### Adstock transformation

Use adstock transformation to take into account lag effects

```python
advertising = [6, 27, 0, 0, 20, 0, 20]

for i in range(1, len(advertising)):
    advertising[i] += advertising[i-1] * 0.5

print(advertising)
```

```sh
>>> [6, 30.0, 15.0, 7.5, 23.75, 11.875, 25.9375]
```

## Timeseries forecasting

### Temporal cross-validation

### Spectral analysis for uncovering recurring patterns
