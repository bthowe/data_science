import sys
import numpy as np
import pandas as pd
from scipy.stats import beta
from scipy.stats import binom

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
    N_groups = int(N / 10)

    beta0 = -1.6
    beta1 = -0.03
    beta2 = 0.6
    beta3 = 1.6

    gamma0 = -3.2
    gamma1 = -0.45
    gamma2 = 0.2
    gamma3 = 3.1

    df = pd.DataFrame(np.random.uniform(-1, 1, size=(N, 3)), columns=['X1', 'X2', 'X3'])
    df['pi_alpha'] = np.exp(beta0 + beta1 * df['X1'] + beta2 * df['X2'] + beta3 * df['X3']) / (1 + np.exp(beta0 + beta1 * df['X1'] + beta2 * df['X2'] + beta3 * df['X3']))
    df['pi_beta'] = np.exp(gamma0 + gamma1 * df['X1'] + gamma2 * df['X2'] + gamma3 * df['X3']) / (1 + np.exp(gamma0 + gamma1 * df['X1'] + gamma2 * df['X2'] + gamma3 * df['X3']))
    print(df['pi_alpha'].describe())
    print(df['pi_beta'].describe())
    df['y'] = beta.rvs(df['pi_alpha'], df['pi_beta'])
    print(df.head())
    sys.exit()
    df.drop(['pi_alpha', 'pi_beta'], 1, inplace=True)
    df['constant'] = 1
    return df

def beta_reg(df, covars, target):
    # http://www.win-vector.com/blog/2014/01/generalized-linear-models-for-predicting-rates/#more-2527

    pandas2ri.activate()
    r = robjects.r
    forecast = importr('betareg')

    rdf = pandas2ri.py2ri(df[[target] + covars])

    print(rdf.head())

    covar_linear_combo = ' + '.join(covars)
    formula = '{0} ~ {1} | {1}'.format(target, covar_linear_combo)
    M = r.betareg(formula, data=df)
    print(r.summary(M))
    print(r.summary(M).rx2('coefficients'))


    # pandas2ri.activate()
    # r = robjects.r
    # forecast = importr('betareg')
    # r.data('GasolineYield', package='betareg')
    # # print(r.sapply(r['GasolineYield'], r.typeof))
    #
    # print(r['GasolineYield'].head())
    # # r.set_seed(52352)  # this doens't work
    #
    # M = r.betareg('yield ~ gravity + pressure + temp | gravity + pressure + temp', data=r['GasolineYield'])
    # print(r.summary(M))
    # print(r.summary(M).rx2('coefficients'))
    #
    #





    # df_train, df_holdout = make_train_holdout('cpa_forecast_data_files/train_data_SC v4.csv')
    # y_train, X_train = make_train_data(df_train)
    # X_score = make_forecast_data(df_holdout)
    #
    # # print X_train.info()
    # # print y_train.info()
    # # sys.exit()
    #
    # covar_lst = X_train.drop(['date'], 1).columns.tolist()
    #
    # # read in df
    # rX_train = pandas2ri.py2ri(X_train[covar_lst])
    # ry_train = r.ts(pandas2ri.py2ri(y_train['target']), start=r.c(2014, 1), frequency=12)
    # rX_score = pandas2ri.py2ri(X_score[covar_lst])
    #
    # # print r.head(ry_train)
    # # print r.sapply(rX_train, r.typeof)
    #
    #
    #
    # print
    # forecast.auto_arima(ry_train, xreg=rX_train)
    # sys.exit()
    # arima_fit = forecast.auto_arima(ry_train, xreg=rX_train)
    # print
    # r.summary(arima_fit)
    #
    # sys.exit()
    # forecast.forecast_checkresiduals(arima_fit)
    #
    # print
    # forecast.forecast(arima_fit, h=36, xreg=rX_score)
    # # print forecast.forecast_fracdiff(arima_fit, xreg=rX_train)

if __name__ == '__main__':
    np.random.seed(2)
    df = artificial_data1()

    print(df['y'].describe()); sys.exit()

    beta_reg(df, ['X1', 'X2', 'X3'], 'y')
    # beta_reg(df, ['constant', 'covarX1', 'covarX2', 'covarX3'], 'y_beta')
    # todo: will it add a constant by default...yes
