# Natural Language Processing (NLP)

## Norvig spelling corrector

Adapted from [here](https://norvig.com/spell-correct.html).

```python
>>> import xam

>>> sentences = [
...     'I madz a mistkze',
...     'I did toi'
... ]

>>> word_counts = {
...     'I': 16,
...     'a': 13,
...     'did': 15,
...     'made': 42,
...     'to': 12,
...     'too': 24,
...     'mistake': 36
... }

>>> alphabet = 'abcdefghijklmnopqrstuvwxyz'
>>> corrector = xam.nlp.NorvigSpellingCorrector(word_counts, alphabet)
>>> corrector = corrector.fit(sentences)
>>> corrector.transform(sentences)
['I made a mistake', 'I did too']

>>> for i, sentence in enumerate(sentences):
...     n_mistakes = corrector.count_sentence_mistakes(sentence)
...     print('Number of mistakes for sentence {}: {}'.format(i, n_mistakes))
Number of mistakes for sentence 0: 2
Number of mistakes for sentence 1: 1

```


## Top-terms classifier

```python
>>> from sklearn.datasets import fetch_20newsgroups
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> import xam

>>> cats = ['alt.atheism', 'comp.windows.x']
>>> newsgroups_train = fetch_20newsgroups(subset='train', categories=cats)
>>> newsgroups_test = fetch_20newsgroups(subset='test', categories=cats)

>>> vectorizer = CountVectorizer(stop_words='english', max_df=0.2)

>>> X_train = vectorizer.fit_transform(newsgroups_train.data)
>>> y_train = newsgroups_train.target

>>> X_test = vectorizer.transform(newsgroups_test.data)
>>> y_test = newsgroups_test.target

>>> clf = xam.nlp.TopTermsClassifier(n_terms=50)
>>> score = clf.fit(X_train.toarray(), y_train).score(X_test.toarray(), y_test)
>>> round(score, 5)
0.95238

```
