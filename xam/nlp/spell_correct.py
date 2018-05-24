from sklearn import base
from sklearn.utils import validation


class NorvigSpellingCorrector(base.BaseEstimator, base.TransformerMixin):
    """Reference: https://norvig.com/spell-correct.html"""

    def __init__(self, word_counts=None, alphabet=None, tokenize=lambda s: s.split()):
        self.word_counts = word_counts
        self.alphabet = alphabet
        self.tokenize = tokenize

    def fit(self, X, y=None):
        self.n_ = sum(self.word_counts.values())
        return self

    def _p(self, word):
        """Prior probability of a word occurring."""
        return self.word_counts.get(word, 0) / self.n_

    def _known(self, words):
        """The subset of words that exist."""
        return set(w for w in words if w in self.word_counts)

    def _candidates(self, word):
        """Generate possible spelling corrections for a word."""
        return (
            self._known([word]) or
            self._known(self._edits1(word)) or
            self._known(self._edits2(word)) or
            [word]
        )

    def _edits1(self, word):
        """All edits that are one edit away from a word."""
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in self.alphabet]
        inserts = [L + c + R for L, R in splits for c in self.alphabet]
        return set(deletes + transposes + replaces + inserts)

    def _edits2(self, word):
        """All edits that are two edits away from a word."""
        return (e2 for e1 in self._edits1(word) for e2 in self._edits1(e1))

    def correct_word(self, word):
        """Most probable spelling correction for a word."""
        return max(self._candidates(word), key=self._p)

    def correct_sentence(self, sentence):
        """Most probable spelling correction for a sentence."""
        return ' '.join(self.correct_word(word) for word in self.tokenize(sentence))

    def count_sentence_mistakes(self, sentence):
        """Count number of spelling mistakes in a sentence."""
        return sum(word != self.correct_word(word) for word in self.tokenize(sentence))

    def transform(self, X):
        validation.check_is_fitted(self, 'n_')
        return [
            self.correct_sentence(sentence)
            for sentence in X
        ]
