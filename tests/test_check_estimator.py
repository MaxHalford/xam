from sklearn.utils.estimator_checks import check_estimator

import xam


# Preprocessing

def test_cycle_transformer():
    assert check_estimator(xam.feature_extraction.CycleTransformer) is None

## Binning

def test_bayesian_blocks_binner():
    assert check_estimator(xam.preprocessing.BayesianBlocksBinner) is None

def test_equal_frequency_binner():
    assert check_estimator(xam.preprocessing.EqualFrequencyBinner) is None

def test_equal_width_binner():
    assert check_estimator(xam.preprocessing.EqualWidthBinner) is None

def test_mdlp_binner():
    assert check_estimator(xam.preprocessing.MDLPBinner) is None


# NLP

# def test_top_terms_classifier():
#     assert check_estimator(xam.nlp.TopTermsClassifier) is None
