from sklearn.utils.estimator_checks import check_estimator

import xam


# Preprocessing

def test_binary_encoding():
    assert check_estimator(xam.preprocessing.BinaryEncoder) is None

def test_cycle_transformer():
    assert check_estimator(xam.preprocessing.CycleTransformer) is None

def test_supervised_imputer():

    # This is needed for going through the same testing hoops as scikit-learn's Imputer
    xam.preprocessing.SupervisedImputer.__name__ = 'Imputer'

    assert check_estimator(xam.preprocessing.SupervisedImputer) is None

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
