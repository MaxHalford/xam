import numpy as np
import pandas as pd
from scipy import stats
from sklearn import feature_selection


def feature_importance(features, target):
    num = features.select_dtypes(include=[float])
    cat = features.select_dtypes(include=[int, bool, object])

    # If the target is numerical then it is a regression task
    if target.dtype == 'float':
        return _regression_importance(num, cat, target)

    # Else it is a classification task
    return _classification_importance(num, cat, target)


def _regression_importance(num, cat, target):

    num_imp = pd.DataFrame(index=num.columns)
    cat_imp = pd.DataFrame(index=cat.columns)

    # Use Pearson correlation for numerical features
    if num_imp.index.size > 0:
        pearson_rs = np.array([stats.pearsonr(feature, target) for _, feature in num.iteritems()])
        num_imp['pearson_r_value'] = pearson_rs[:, 0]
        num_imp['pearson_r_p'] = pearson_rs[:, 1]

    # Use mutual information for categorical features
    if cat_imp.index.size > 0:
        mut_inf = feature_selection.mutual_info_regression(cat, target, discrete_features=True)
        cat_imp['mutual_information'] = mut_inf

    return num_imp, cat_imp


def _classification_importance(num, cat, target):

    num_imp = pd.DataFrame(index=num.columns)
    cat_imp = pd.DataFrame(index=cat.columns)

    # Use mutual information for numerical features
    if num_imp.index.size > 0:
        mut_inf = feature_selection.mutual_info_classif(num, target, discrete_features=False)
        num_imp['mutual_information'] = mut_inf

    # Use chi-square score for categorical features
    if cat_imp.index.size > 0:
        chi2 = feature_selection.chi2(cat, target)
        cat_imp['chi2_value'], cat_imp['chi2_p'] = chi2[0], chi2[1]

    return num_imp, cat_imp
