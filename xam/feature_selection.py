from collections import defaultdict
import math

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import feature_selection


def cramers_v_stat(confusion_matrix):
    """Calculate Cramérs V statistic for categorial-categorial association."""
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return math.sqrt(phi2 / min((r-1), (k-1)))


def cramers_v_corrected_stat(confusion_matrix):
    """Calculate Cramérs V statistic for categorial-categorial association.

    Uses correction from Bergsma and Wicher, Journal of the Korean Statistical
    Society 42 (2013): 323-328.
    """
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2_corr = max(0, phi2 - ((k-1)*(r-1)) / (n-1))
    r_corr = r - ((r-1)**2) / (n-1)
    k_corr = k - ((k-1)**2) / (n-1)
    return math.sqrt(phi2_corr / min((r_corr-1), (k_corr-1)))


def feature_importance_classification(features, target, n_neighbors=3, random_state=None):

    cont = features.select_dtypes(include=[np.floating])
    disc = features.select_dtypes(include=[np.integer, np.bool])

    cont_imp = pd.DataFrame(index=cont.columns)
    disc_imp = pd.DataFrame(index=disc.columns)

    # Continuous features
    if cont_imp.index.size > 0:

        # F-test
        f_test = feature_selection.f_classif(cont, target)
        cont_imp['f_statistic'] = f_test[0]
        cont_imp['f_p_value'] = f_test[1]

        # Mutual information
        mut_inf = feature_selection.mutual_info_classif(cont, target, discrete_features=False,
                                                        n_neighbors=n_neighbors,
                                                        random_state=random_state)
        cont_imp['mutual_information'] = mut_inf

    # Discrete features
    if disc_imp.index.size > 0:

        # Chi²-test
        chi2_tests = defaultdict(dict)

        for feature in disc.columns:
            cont = pd.crosstab(disc[feature], target)
            statistic, p_value, _, _ = stats.chi2_contingency(cont)
            chi2_tests[feature]['chi2_statistic'] = statistic
            chi2_tests[feature]['chi2_p_value'] = p_value

        chi2_tests_df = pd.DataFrame.from_dict(chi2_tests, orient='index')
        disc_imp['chi2_statistic'] = chi2_tests_df['chi2_statistic']
        disc_imp['chi2_p_value'] = chi2_tests_df['chi2_p_value']

        # Cramér's V (corrected)
        disc_imp['cramers_v'] = [
            cramers_v_corrected_stat(pd.crosstab(feature, target).values)
            for _, feature in disc.iteritems()
        ]

        # Mutual information
        mut_inf = feature_selection.mutual_info_classif(disc, target, discrete_features=True,
                                                        n_neighbors=n_neighbors,
                                                        random_state=random_state)
        disc_imp['mutual_information'] = mut_inf

    return cont_imp, disc_imp


def feature_importance_regression(features, target, n_neighbors=3, random_state=None):

    cont = features.select_dtypes(include=[np.floating])
    disc = features.select_dtypes(include=[np.integer, np.bool])

    cont_imp = pd.DataFrame(index=cont.columns)
    disc_imp = pd.DataFrame(index=disc.columns)

    # Continuous features
    if cont_imp.index.size > 0:

        # Pearson correlation
        pearson = np.array([stats.pearsonr(feature, target) for _, feature in cont.iteritems()])
        cont_imp['pearson_r'] = pearson[:, 0]
        cont_imp['pearson_r_p_value'] = pearson[:, 1]

        # Mutual information
        mut_inf = feature_selection.mutual_info_regression(cont, target, discrete_features=False,
                                                           n_neighbors=n_neighbors,
                                                           random_state=random_state)
        cont_imp['mutual_information'] = mut_inf

    # Discrete features
    if disc_imp.index.size > 0:

        # F-test
        f_tests = defaultdict(dict)

        for feature in disc.columns:
            groups = [target[idxs] for idxs in disc.groupby(feature).groups.values()]
            statistic, p_value = stats.f_oneway(*groups)
            f_tests[feature]['f_statistic'] = statistic
            f_tests[feature]['f_p_value'] = p_value

        f_tests_df = pd.DataFrame.from_dict(f_tests, orient='index')
        disc_imp['f_statistic'] = f_tests_df['f_statistic']
        disc_imp['f_p_value'] = f_tests_df['f_p_value']

        # Mutual information
        mut_inf = feature_selection.mutual_info_regression(disc, target, discrete_features=True,
                                                           n_neighbors=n_neighbors,
                                                           random_state=random_state)
        disc_imp['mutual_information'] = mut_inf

    return cont_imp, disc_imp
