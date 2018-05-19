import unittest

from sklearn.utils.estimator_checks import check_estimator

import xam


class TestBayesianBlocksBinner(unittest.TestCase):

    def test_check_estimator(self):
        assert check_estimator(xam.preprocessing.BayesianBlocksBinner) is None


class TestEqualFrequencyBinner(unittest.TestCase):

    def test_check_estimator(self):
        assert check_estimator(xam.preprocessing.EqualFrequencyBinner) is None


class TestEqualWidthBinner(unittest.TestCase):

    def test_check_estimator(self):
        assert check_estimator(xam.preprocessing.EqualWidthBinner) is None


class TestMDLPBinner(unittest.TestCase):

    def test_check_estimator(self):
        assert check_estimator(xam.preprocessing.MDLPBinner) is None
