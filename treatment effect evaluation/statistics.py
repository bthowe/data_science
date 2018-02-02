import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def abs_ave_by_treatment(df, Y, W):
    """Calculates the absolute value of the difference in average outcomes by treatment status. This test statistic is
    relatively attractive if the most interesting alternative hypothesis corresponds to an additive treatment effect,
    and the frequency distributions of Y_i(0) and Y_i(1) have few outliers.

    df: the dataset
    Y: the outcome variable
    W: the treatment assignment
    """
    return np.abs(df[Y].loc[df[W] == 1].mean() - df[Y].loc[df[W] == 0].mean())

def abs_ave_by_treatment(df, f, Y, W):
    """Calculates the absolute value of the difference in average transformed outcomes by treatment status. This test
    statistic is relatively attractive if a plausible alternative hypothesis corresponds to an additive treatment effect
    after such a transformation (e.g., the natural logorithm associated with a constant multiplicative effect of the
    treatment). Sensible if the raw data have skewed distributions.

    df: the dataset
    f: function that transforms the outcomes
    Y: the outcome variable
    W: the treatment assignment
    """
    return np.abs(f(df[Y].loc[df[W] == 1]).mean() - f(df[Y].loc[df[W] == 0]).mean())

def diff_quantile_by_treatment(df, q, Y, W):
    """Calculates the absolute value of the difference in a specified quantile of the empirical distributions by
    treatment status. These give robust estimates of location (i.e., are not sensitive to outliers).

    df: the dataset
    q: the desired quantile
    Y: the outcome variable
    W: the treatment assignment
    """
    return np.abs(df[Y].loc[df[W] == 1].quantile(q) - df[Y].loc[df[W] == 0].quantile(q))

def t_stat(df, Y, W):
    """Calculates the t-statistic for the test of the null hypothesis of equal means, with unequal variances in the two
    groups.

    df: the dataset
    Y: the outcome variable
    W: the treatment assignment
    """
    Y_t = df[Y].loc[df[W] == 1].mean()
    Y_c = df[Y].loc[df[W] == 0].mean()
    N_t = len(df[Y].loc[df[W] == 1])
    N_c = len(df[Y].loc[df[W] == 0])
    s_t = (Y_c - df[Y].loc[df[W] == 1]) ** 2 / (N_t - 1)
    s_c = (Y_c - df[Y].loc[df[W] == 0]) ** 2 / (N_c - 1)
    return np.abs(
        (Y_t - Y_c) / np.sqrt((s_t ** 2 / N_t) + (s_c ** 2 / N_c))
    )

def abs_ave_rank_by_treatment(df, Y, W):
    """Calculates the absolute value of the difference in average normalized rank of the outcomes by treatment status.
    This test statistic is relatively attractive when the raw outcomes have a distribution with a substantial number of
    outliers. Rank is defined as the count of all observations less than or equal in magnitude to a given point. Ties
    are resolved by averaging the possible ranks for the tied observations.

    df: the dataset
    Y: the outcome variable
    W: the treatment assignment
    """
    N = len(df)
    df['normalized_rank'] = df[Y].rank() - (N + 1) / 2
    R_t = df['normalized_rank'].loc[df[W] == 1].mean()
    R_c = df['normalized_rank'].loc[df[W] == 0].mean()
    return np.abs(R_t - R_c)

def model_based_stat(theta_t, theta_c, stat_func):
    theta = pd.DataFrame([[theta_t, 1], [theta_c, 0]], columns=['Y', 'W'])
    # todo: pass theta into stat_func along with a dic of parameters I need to also specify in the function definition
    pass

def kol_smir_stat(df, Y, W):
    # todo: description here
    ks_2samp(df[Y].loc[df[W] == 1], df[Y].loc[df[W] == 0])


