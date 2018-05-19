import unittest

from sklearn.utils.estimator_checks import check_estimator

import xam


class TestCycleTransformer(unittest.TestCase):

    def test_check_estimator(self):
        assert check_estimator(xam.feature_extraction.CycleTransformer) is None
