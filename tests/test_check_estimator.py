from sklearn.utils.estimator_checks import check_estimator

import xam


def test_mdlp_binner():
    assert check_estimator(xam.preprocessing.MDLPBinner) is None

#def test_supervised_imputer():
#    assert check_estimator(xam.preprocessing.SupervisedImputer) is None
