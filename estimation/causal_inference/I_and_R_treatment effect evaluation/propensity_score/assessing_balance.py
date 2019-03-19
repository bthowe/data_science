import sys
import numpy as np
import pandas as pd
from scipy.stats import f
from scipy.stats import norm
from itertools import product
import matplotlib.pyplot as plt
import pandas.util.testing as tm

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

np.random.seed(444)


def data_create():
    tm.N, tm.K = 300, 5
    df = tm.makeDataFrame()
    df['W'] = np.random.choice([0, 1], tm.N)
    df['block'] = np.random.choice(range(5), tm.N)
    return df


def _X_temp_create(X, block):
    X_temp_c = X.query('(block == {0}) and (W == 0)'.format(block))
    X_temp_t = X.query('(block == {0}) and (W == 1)'.format(block))
    N_c = X_temp_c.shape[0]
    N_t = X_temp_t.shape[0]
    return X_temp_c, X_temp_t, N_c, N_t

def _variance(X, block, column):
    X_temp_c, X_temp_t, N_c, N_t = _X_temp_create(X, block)
    s2 = (1 / (N_c - 2)) * (((X_temp_c[column] - X_temp_c[column].mean()) ** 2).sum() + ((X_temp_t[column] - X_temp_t[column].mean()) ** 2).sum())
    return s2 * ((1 / N_c) + (1 / N_t))

def _mean(X, block, column):
    X_temp_c, X_temp_t, N_c, N_t = _X_temp_create(X, block)
    return X_temp_t[column].mean() - X_temp_c[column].mean()

def _N_weight(X, block):
    return X.query('block == {0}'.format(block)).shape[0] / X.shape[0]


def covariate_by_strata(X):
    covar_score = {}
    for col in [col for col in X.columns if col not in ['W', 'block']]:
        pseudo_ce = 0
        sampling_var = 0
        for block in X['block'].unique():
            pseudo_ce += _mean(X, block, col) * _N_weight(X, block)
            sampling_var += _variance(X, block, col) * (_N_weight(X, block) ** 2)
        covar_score[col] = pseudo_ce / np.sqrt(sampling_var)
    return covar_score


def SSR(X, column, type):
    SSR = 0
    for block in X['block'].unique():
        X_temp_c, X_temp_t, N_c, N_t = _X_temp_create(X, block)

        if type == 'restricted':
            a_r = (N_c / (N_c + N_t)) * X_temp_c[column].mean() + (N_t / (N_c + N_t)) * X_temp_t[column].mean()
            SSR += ((X_temp_c[column] - a_r) ** 2).sum() + ((X_temp_t[column] - a_r) ** 2).sum()
            # SSR += ((X.query('block == {0}'.format(block))[column] - a_r) ** 2).sum()
        else:
            a_ur = X_temp_c[column].mean()
            t = _mean(X, block, column)
            SSR += ((X_temp_c[column] - a_ur) ** 2).sum() + ((X_temp_t[column] - a_ur - t) ** 2).sum()

    return SSR

def z_score_plot(scores):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(scores)
    plt.show()


def covariate_within_block(X, plot=True):
    """Finding large positive values suggests that the covariates are not balanced within the strata"""
    N = X.shape[0]
    J = len(X['block'].unique())

    covar_score = {}
    for col in [col for col in X.columns if col not in ['W', 'block']]:
        F_stat = ((SSR(X, col, 'restricted') - SSR(X, col, 'unrestricted')) / J) / (SSR(X, col, 'unrestricted') / (N - 2 * J))
        pscore = f.cdf(F_stat, J, (N - 2 * J))
        covar_score[col] = norm.ppf(pscore)
    if plot:
        z_score_plot(covar_score.values())
    return covar_score


def covariate_and_block(X, print=True):
    """If the covariates are well balanced, we would expect to find the absolute values of the z-values to be
    concentrated toward smaller (less significant) values relative to a normal distribution."""
    covar_score = {}
    for col, block in product([col for col in X.columns if col not in ['W', 'block']], X['block'].unique()):
        X_temp_c, X_temp_t, N_c, N_t = _X_temp_create(X, block)
        X_temp_c[col].mean() - X_temp_c[col].mean()
        covar_score['{0}, {1}'.format(col, block)] = _mean(X, block, col) / np.sqrt(_variance(X, block, col) * ((1 / N_c) + (1 / N_t)))
    if print:
        Q_Q_plot(list(covar_score.values()))
    return covar_score

def Q_Q_plot(df):
    """Plot randomly generated normally distributed observations (x-axis) against observations from the empirical
    distribution. If the empirical distribution is normally distributed the points should lie roughly along the
    diagonal."""
    n = len(df)
    empirical_lst = np.sort(df)
    norm_lst = np.sort(norm.rvs(size=n))

    min_val = min(min(empirical_lst), min(norm_lst))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.linspace(min_val, -min_val, 100), np.linspace(min_val, -min_val, 100))
    ax.scatter(norm_lst, empirical_lst)
    plt.title('Q-Q Plot')
    plt.show()
    sys.exit()


def main():
    df = data_create()

    ## assessing global balance for each covariate across strata
    print(covariate_by_strata(df))

    ## assessing balance for each covariate within all blocks
    print(covariate_within_block(df))

    ## assessing balance within strata for each covariate
    print(covariate_and_block(df))

if __name__ == '__main__':
    main()
