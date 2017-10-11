### Plotting

**Latex style figures**

```python
>>> from xam import latex  # Has to be imported before matplotlib.pyplot
>>> import numpy as np

>>> fig, ax = latex.new_fig(width=0.8)

>>> x = np.arange(-2, 2, 0.03)
>>> y1 = 1 / (1 + np.exp(-x))
>>> y2 = np.tanh(x)
>>> y3 = np.arctan(x)
>>> y4 = x * (x > 0)

>>> plot = ax.plot(x, y1, label='Logistic sigmoid')
>>> plot = ax.plot(x, y2, label='Hyperbolic tangent')
>>> plot = ax.plot(x, y3, label='Inverse tangent')
>>> plot = ax.plot(x, y4, label='Rectified linear unit (ReLU)')

>>> x_label = ax.set_xlabel(r'$x$')
>>> y_label = ax.set_ylabel(r'$y$')
>>> title = ax.set_title('A few common activation functions')
>>> ax.grid(linewidth=0.5)
>>> legend = ax.legend(loc='upper left', framealpha=1)

latex.save_fig('figures/latex_example')

```

<div align="center">
  <img src="figures/latex_example.png" width="80%">
</div>
