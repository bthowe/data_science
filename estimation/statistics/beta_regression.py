import sys
import numpy as np
import pandas as pd
from scipy.stats import beta
import matplotlib.pyplot as plt

import os
os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'
# todo: add export path to .bash_profile

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def artificial_data1():
    N = 1000

    beta0 = -1.6
    beta1 = -0.03
    beta2 = 0.6
    beta3 = 1.6

    gamma0 = 3.2
    gamma1 = 0.45
    gamma2 = 0.2
    gamma3 = 3.1

    df = pd.DataFrame(np.random.uniform(0, 2, size=(N, 3)), columns=['X1', 'X2', 'X3'])
    df['pi_mu'] = np.exp(beta0 + beta1 * df['X1'] + beta2 * df['X2'] + beta3 * df['X3']) / (1 + np.exp(beta0 + beta1 * df['X1'] + beta2 * df['X2'] + beta3 * df['X3']))  # logit link
    df['pi_phi'] = np.log(gamma0 + gamma1 * df['X1'] + gamma2 * df['X2'] + gamma3 * df['X3'])  # log link
    # todo: this doesn't seem to be correct for some reason, given the output. It looks like the regression estimates the betas well but not the gammas.

    df['a'] = df['pi_mu'] * df['pi_phi']
    df['b'] = df['pi_phi'] * (1 - df['pi_mu'])

    df['y'] = beta.rvs(df['a'], df['b'])
    df.drop(['pi_mu', 'pi_phi', 'a', 'b'], 1, inplace=True)
    return df


def _prediction_uncertainty_range_plot(y_pred, y_true, y_precision):
    mn = min(np.min(y_pred), np.min(y_true))
    mx = max(np.max(y_pred), np.max(y_true))
    std = np.std(y_pred) / 2

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.linspace(mn, mx, 100), np.linspace(mn, mx, 100), color='black')
    ax.scatter(y_pred, y_true)
    ax.errorbar(y_pred, y_true, fmt='o', xerr=y_precision)
    ax.set_ylim(mn - std, mx + std)
    ax.set_xlim(mn - std, mx + std)
    plt.show()

def beta_reg(df, target, plot=True):
    """Use beta regression when estimating non-frequency rates and not when estimating probabilities or frequencies.
    Remember that the outcome value needs to be between zero and one."""
    pandas2ri.activate()
    r = robjects.r
    importr('betareg')

    covars = [covar for covar in df.columns.tolist() if covar != target]
    print(covars)

    covar_linear_combo = ' + '.join(covars)
    formula = '{0} ~ {1} | {1}'.format(target, covar_linear_combo)
    M = r.betareg(formula, data=df, link='logit', link_phi='log')  # link_phi can be 'indentity', 'log', or 'sqrt'; adds a constant by default

    print(r.summary(M))
    print(r.summary(M).rx2('coefficients'))

    _prediction_uncertainty_range_plot(r.predict(M, newdata=df), df[target], np.sqrt(r.predict(M, newdata=df, type='variance')))


def test_beta_reg():
    """Related to http://www.win-vector.com/blog/2014/01/generalized-linear-models-for-predicting-rates/#more-2527"""
    pandas2ri.activate()
    r = robjects.r
    forecast = importr('betareg')
    r.data('GasolineYield', package='betareg')
    # print(r['GasolineYield'].head())
    # print(r.sapply(r['GasolineYield'], r.typeof))

    print(r['GasolineYield'].head())
    # r.set_seed(52352)  # this doens't work

    M = r.betareg('yield ~ gravity + pressure + temp | gravity + pressure + temp', data=r['GasolineYield'])
    print(r.summary(M))
    print(r.summary(M).rx2('coefficients'))

    _prediction_uncertainty_range_plot(r.predict(M, newdata=r['GasolineYield']), r['GasolineYield']['yield'], np.sqrt(r.predict(M, newdata=r['GasolineYield'], type='variance')))

    # rdf = pandas2ri.py2ri(df[[target] + covars])

if __name__ == '__main__':
    np.random.seed(2)
    df = artificial_data1()

    beta_reg(df, 'y')
    test_beta_reg()
