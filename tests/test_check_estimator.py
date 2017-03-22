from sklearn.utils.estimator_checks import check_estimator

from xam.preprocessing import MDLPBinner
from xam.preprocessing import EqualWidthBinner


def test_mdlp_binner():
    assert check_estimator(MDLPBinner) is None
