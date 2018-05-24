# xam [![Build Status](https://travis-ci.org/MaxHalford/xam.svg?branch=master)](https://travis-ci.org/MaxHalford/xam)

xam is my personal data science and machine learning toolbox. It is written in Python 3 and stands on the shoulders of giants (mainly [pandas](https://pandas.pydata.org/) and [scikit-learn](http://scikit-learn.org/)).

## Installation

- [Install Anaconda for Python 3.x >= 3.5](https://www.continuum.io/downloads)
- Run `pip install git+https://github.com/MaxHalford/xam --upgrade` in a terminal

:warning: Because xam is a ***personal*** toolkit, the `--upgrade` flag will install the latest releases of each dependency (scipy, pandas etc.).

## Table of contents

Usage example is available in the [docs](docs) folder. Each example is tested with [doctest](https://pymotw.com/2/doctest/).

- [Clustering](docs/clustering.md)
  - [Cross-chain algorithm](docs/clustering.md#cross-chain-algorithm)
- [Ensembling](docs/ensembling.md)
  - [Groupby model](docs/ensembling.md#groupby-model)
  - [Stacking](docs/ensembling.md#stacking)
- [Exploratory data analysis (EDA)](docs/eda.md)
  - [Feature importance](docs/eda.md#feature-importance)
- [Feature extraction](docs/feature-extraction.md)
  - [Combining features](docs/feature-extraction.md#combining-features)
  - [Cyclic features](docs/feature-extraction.md#cyclic-features)
  - [K-fold target encoding](docs/feature-extraction.md#k-fold-target-encoding)
  - [Smooth target encoding](docs/feature-extraction.md#smooth-target-encoding)
- [Linear models](docs/linear-models.md)
  - [AUC regressor](docs/linear-models.md#auc-regressor)
- [Model selection](docs/model-selection.md)
  - [Ordered cross-validation](docs/model-selection.md#ordered-cross-validation)
- [Natural Language Processing (NLP)](docs/nlp.md)
  - [Norvig spelling corrector](docs/nlp.md#norvig-spelling-corrector)
  - [Top-terms classifier](docs/nlp.md#top-terms-classifier)
- [Pipeline](docs/pipeline.md)
  - [Column selection](docs/pipeline.md#column-selection)
  - [Series transformer](docs/pipeline.md#series-transformer)
  - [DataFrame transformer](docs/pipeline.md#dataframe-transformer)
  - [Lambda transformer](docs/pipeline.md#lambda-transformer)
- [Plotting](docs/plotting.md)
  - [Latex style figures](docs/plotting.md#latex-style-figures)
- [Preprocessing](docs/preprocessing.md)
  - [Binning](docs/preprocessing.md#binning)
  - [Groupby transformer](docs/preprocessing.md#groupby-transformer)
  - [Resampling](docs/preprocessing.md#resampling)
- [Time series analysis (TSA)](docs/tsa.md)
  - [Exponentially weighted average](docs/tsa.md#ewm-optimization)
  - [Exponential smoothing](docs/tsa.md#exponential-smoothing)
  - [Frequency average forecasting](docs/tsa.md#frequency-average-forecasting)
- [Various](docs/various.md)
  - [Datetime range](docs/various.md#datetime-range)
  - [Next day of the week](docs/various.md#next-day-of-the-week)
  - [Subsequence lengths](docs/various.md#subsequence-lengths)
  - [DataFrame to Vowpal Wabbit](docs/various.md#dataFrame-to-vowpal-wabbit)

## Other Python data science and machine learning toolkits

- [fastai/fastai](https://github.com/fastai/fastai)
- [Laurae2/Laurae](https://github.com/Laurae2/Laurae)
- [rasbt/mlxtend](https://github.com/rasbt/mlxtend)
- [reiinakano/scikit-plot](https://github.com/reiinakano/scikit-plot)
- [scikit-learn-contrib](https://github.com/scikit-learn-contrib)
- [zygmuntz/phraug2](https://github.com/zygmuntz/phraug2)

## License

The MIT License (MIT). Please see the [license file](LICENSE) for more information.
