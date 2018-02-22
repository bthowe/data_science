import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import binom
from statsmodels.genmod.families import links as L

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

    df = pd.DataFrame(np.random.uniform(-1, 1, size=(N, 3)), columns=['X1', 'X2', 'X3'])
    df['pi_x'] = np.exp(beta0 + beta1 * df['X1'] + beta2 * df['X2'] + beta3 * df['X3']) / (1 + np.exp(beta0 + beta1 * df['X1'] + beta2 * df['X2'] + beta3 * df['X3']))
    df['y'] = binom.rvs(1, df['pi_x'])
    df['constant'] = 1
    df['group'] = np.random.choice(range(N_groups), size=(N, 1))

    df['covarX1'] = 0
    df['covarX2'] = 0
    df['covarX3'] = 0
    for g in range(N_groups):
        df.loc[df['group'] == g, 'covarX1'] = df['X1'].loc[df['group'] == g].mean()
        df.loc[df['group'] == g, 'covarX2'] = df['X2'].loc[df['group'] == g].mean()
        df.loc[df['group'] == g, 'covarX3'] = df['X3'].loc[df['group'] == g].mean()

    return df

def artificial_data2():
    # todo: how do the models do given a different data generation process?
    pass


def log_reg(df, covars, target):
    # logit = sm.Logit(df[target], df[covars])
    # print(logit.fit().params)

    # or

    gamma_model = sm.GLM(df[target], df[covars], family=sm.families.Binomial())
    gamma_results = gamma_model.fit()
    print(gamma_results.summary())


def binomial_reg(df, covars, target):
    """
    very clear explanation: https://stats.stackexchange.com/questions/144121/logistic-regression-bernoulli-vs-binomial-response-variables?rq=1

    links = [L.logit, L.probit, L.cauchy, L.log, L.cloglog, L.identity]
    sm.families.Binomial(link=L.probit)
    default is L.logit

    """
    gamma_model = sm.GLM(df[target], df[covars], family=sm.families.Binomial(link=L.probit))
    gamma_results = gamma_model.fit()
    print(gamma_results.summary())

def frac_log_reg(df, covars, target):
    """
    cov_type='HC0' employs the sandwich (i.e., heteroskedastic consistent) covariance
    While not mentioned explicityly, I'm assuming the estimation is done using quasi-maximum likelihood since the
    target is not an integer and so the usual logistic regression likelihood function cannot be used. Maybe I should
     scour the code
    """
    logit = sm.Logit(df[target], df[covars])
    print(logit.fit(cov_type='HC0').params)


if __name__ == '__main__':
    np.random.seed(2)
    df = artificial_data1()

    log_reg(df, ['constant', 'X1', 'X2', 'X3'], 'y')
    log_reg(df, ['constant', 'covarX1', 'covarX2', 'covarX3'], 'y')

    df = df.groupby(df['group']).agg({'y': ['sum', 'count'], 'covarX1': 'mean', 'covarX2': 'mean', 'covarX3': 'mean'}).reset_index(drop=True)
    df.columns = ['y_success', 'y_count', 'covarX1', 'covarX2', 'covarX3']
    df['constant'] = 1
    df['y_failure'] = df['y_count'] - df['y_success']
    df['y_frac'] = df['y_success'] / df['y_count']
    df.drop('y_count', 1, inplace=True)

    # print(df.head())

    binomial_reg(df, ['constant', 'covarX1', 'covarX2', 'covarX3'], ['y_success', 'y_failure'])
    frac_log_reg(df, ['constant', 'covarX1', 'covarX2', 'covarX3'], 'y_frac')
