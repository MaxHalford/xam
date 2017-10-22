# Pipeline

## Column selection

Transformer that extracts one or more columns from a dataframe; is useful for applying a Transformer on a subset of features in a pipeline.

```python
>>> import pandas as pd
>>> import xam

>>> df = pd.DataFrame({'a': [1, 1, 1], 'b': [2, 2, 2], 'c': [3, 3, 3]})

>>> xam.pipeline.ColumnSelector('a').fit_transform(df)
0    1
1    1
2    1
Name: a, dtype: int64

>>> xam.pipeline.ColumnSelector(['b', 'c']).fit_transform(df)
   b  c
0  2  3
1  2  3
2  2  3

```


## Series transformer

Applies a function to each value in series.

```python
>>> import pandas as pd
>>> from sklearn.pipeline import Pipeline
>>> from xam.pipeline import ColumnSelector
>>> from xam.pipeline import SeriesTransformer

>>> df = pd.DataFrame({'a': [1, 1, 1], 'b': [2, 2, 2]})

>>> pipeline = Pipeline([
...    ('extract', ColumnSelector('a')),
...    ('transform', SeriesTransformer(lambda x: 2 * x))
... ])

>>> pipeline.fit_transform(df)
0    2
1    2
2    2
Name: a, dtype: int64

```


## DataFrame transformer

By design scikit-learn Transformers output numpy nd-arrays, the `ToDataFrameTransformer` can be used in a pipeline to return pandas dataframes if needed.

```python
>>> import pandas as pd
>>> from sklearn.pipeline import Pipeline
>>> from xam.pipeline import ColumnSelector
>>> from xam.pipeline import SeriesTransformer
>>> from xam.pipeline import ToDataFrameTransformer

>>> df = pd.DataFrame({'a': [1, 1, 1], 'b': [2, 2, 2]})

>>> pipeline = Pipeline([
...    ('extract', ColumnSelector('a')),
...    ('transform', SeriesTransformer(lambda x: 2 * x)),
...    ('dataframe', ToDataFrameTransformer())
... ])

>>> pipeline.fit_transform(df)
   a
0  2
1  2
2  2

```


## Lambda transformer

Will apply a function to the input; this transformer can potentially do anything but you have to keep track of your inputs and outputs. Alternatively you can use scikit-learn's [`FunctionTransformer`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html) but this only works for numpy arrays, not pandas dataframes.

```python
>>> import pandas as pd
>>> import xam

>>> df = pd.DataFrame({'one': ['a', 'a', 'a'], 'two': ['c', 'a', 'c']})

>>> def has_one_c(dataframe):
...    return (dataframe['one'] == 'c') | (dataframe['two'] == 'c')

>>> xam.pipeline.LambdaTransfomer(has_one_c).fit_transform(df)
0     True
1    False
2     True
dtype: bool

```
