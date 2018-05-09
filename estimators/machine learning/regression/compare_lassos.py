import sys
import numpy as np
import pandas as pd
from scipy.stats import binom
import matplotlib.pyplot as plt
from sklearn.linear_model import MultiTaskLasso, Lasso

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


rng = np.random.RandomState(42)

def data_create():
    N = 1000

    beta0 = -1.6
    beta1 = -0.03
    beta2 = 0.6
    beta3 = 1.6

    df = pd.DataFrame(np.random.uniform(-1, 1, size=(N, 3)), columns=['X1', 'X2', 'X3'])
    df['mu'] = beta0 + beta1 * df['X1'] + beta2 * df['X2'] + beta3 * df['X3']
    df['y1'] = df['mu'] + np.random.normal(0, 1, size=(N,))
    df['y2'] = df['mu'] + 1 + np.random.normal(0, 1, size=(N,))
    df['y3'] = df['mu'] + .5 + np.random.normal(0, 1, size=(N,))

    df['X0'] = 1
    return df

def multi_task_lasso(df):
    X = df[['X0', 'X1', 'X2', 'X3']]
    Y = df[['y1', 'y2', 'y3']]

    # coef_lasso_ = np.array([Lasso(alpha=0.5).fit(X, y).coef_ for y in Y.T])



    print(np.array([Lasso(fit_intercept=False, alpha=0.05).fit(X, Y[y]).coef_ for y in Y]))

    print(MultiTaskLasso(fit_intercept=False, alpha=0.05).fit(X, Y).coef_)

if __name__ == '__main__':
    df = data_create().pipe(multi_task_lasso)


# todo: gridsearch
# todo: compare fits and coefficients
# todo: maybe different data would be better.

