from sklearn import metrics


def temporal_train_test_split(series, train_until):
    train_indexes = series.index <= train_until
    train = series[train_indexes]
    test = series[~train_indexes]
    return train, test


def temporal_train_test_score(forecaster, series, train_until, metric=metrics.mean_squared_error):
    train, test = temporal_train_test_split(series, train_until)
    forecaster.fit(train)
    pred = forecaster.predict(test.index)
    score = metric(test, pred)
    return score
