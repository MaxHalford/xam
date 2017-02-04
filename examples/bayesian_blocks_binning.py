import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from xam.binning import BayesianBlocksBinner


np.random.seed(0)

x = np.concatenate([
    stats.cauchy(-5, 1.8).rvs(500),
    stats.cauchy(-4, 0.8).rvs(2000),
    stats.cauchy(-1, 0.3).rvs(500),
    stats.cauchy(2, 0.8).rvs(1000),
    stats.cauchy(4, 1.5).rvs(500)
])
x = x[(x > -15) & (x < 15)]

binner = BayesianBlocksBinner()
binner.fit(x.reshape(-1, 1))

h1 = plt.hist(
    x,
    bins=200,
    histtype='stepfilled',
    alpha=0.4,
    normed=True
)

bins = np.concatenate([
    [np.min(x)],
    binner.cut_points_[0],
    [np.max(x)]
])

h2 = plt.hist(
    x,
    bins=bins,
    color='black',
    histtype='step',
    normed=True
)

plt.show()
