import numpy as np
from scipy import optimize
from sklearn import metrics


def _calc_shifted_ewm(series, shift, alpha, adjust=True):
    return series.shift(shift).ewm(alpha=alpha, adjust=adjust).mean()


def calc_optimized_ewm(series, shift=1, metric=metrics.mean_squared_error, adjust=False, eps=10e-5, **kwargs):

    def f(alpha):
        shifted_ewm = _calc_shifted_ewm(
            series=series,
            shift=shift,
            alpha=min(max(alpha, 0), 1),
            adjust=adjust
        )
        corr = metric(series[shift:], shifted_ewm[shift:])
        return corr

    res = optimize.differential_evolution(func=f, bounds=[(0 + eps, 1 - eps)], **kwargs)

    return _calc_shifted_ewm(series=series, shift=shift, alpha=res['x'][0], adjust=adjust)
