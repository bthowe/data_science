import sys
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

days = 30
budget_uma = 2.5 * (10 ** 6) / (30 / days)

def ltv_f(spend):
    f = [
        lambda x: (4 * x - (.000025 * (x ** 2))),
        lambda x: (4.5 * x - (.0000275 * (x ** 2))),
        lambda x: (5 * x - (.00003 * (x ** 2))),
        lambda x: (5.5 * x - (.0000325 * (x ** 2)))
    ]

    x_out = []
    num_days = int(len(spend) / 4)
    for i, func in enumerate(f):
        ind = i * num_days
        x_out = np.concatenate((x_out, func(spend[ind: ind + num_days])))
    return -x_out.sum()

def solver_mini():
    initial_spend = np.random.uniform(0, 20000, size=(days * 4,))

    cons = (
        {'type': 'eq', 'fun': lambda x: budget_uma - x.sum()},
        {'type': 'ineq', 'fun': lambda x: x},
    )

    res = minimize(ltv_f, initial_spend, constraints=cons, options={'eps': 0.01})  #, method='SLSQP', tol=0.000001, options={'disp': True, 'ftol': 0.00001, 'eps': 0.01})
    print(res)
    print('objective_value: {}'.format(res['fun']))
    print('sum of spend: {}'.format(res['x'].sum()))
    # sys.exit()
    return res['x']

def plot_allocation(x_opt, x_solver):
    f = [
        lambda x: (4 * x - (.000025 * (x ** 2))),
        lambda x: (4.5 * x - (.0000275 * (x ** 2))),
        lambda x: (5 * x - (.00003 * (x ** 2))),
        lambda x: (5.5 * x - (.0000325 * (x ** 2)))
    ]

    x_out = []
    num_days = int(len(x_solver) / 4)
    for i, func in enumerate(f):
        ind = i * num_days
        x_out = np.concatenate((x_out, func(x_solver[ind: ind + num_days])))

    c = ['cornflowerblue'] * num_days + ['seagreen'] * num_days + ['sandybrown'] * num_days + ['firebrick'] * num_days

    x = np.arange(0, 40000)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, f[0](x), color='cornflowerblue')
    ax.plot(x, f[1](x), color='seagreen')
    ax.plot(x, f[2](x), color='sandybrown')
    ax.plot(x, f[3](x), color='firebrick')

    ax.scatter(x_solver, x_out, color=c)

    ax.scatter(x_opt[0], f[0](x_opt[0]), color='k', alpha=.3)
    ax.scatter(x_opt[1], f[1](x_opt[1]), color='k', alpha=.3)
    ax.scatter(x_opt[2], f[2](x_opt[2]), color='k', alpha=.3)
    ax.scatter(x_opt[3], f[3](x_opt[3]), color='k', alpha=.3)

    plt.show()

def real_solver():
    c = 2.5 * (10 ** 6) / 30
    a1, b1 = 4, .000025
    a2, b2 = 4.5, .0000275
    a3, b3 = 5, .00003
    a4, b4 = 5.5, .0000325

    x1 = (((a1 - a2) / (2 * b2)) + ((a1 - a3) / (2 * b3)) + ((a1 - a4) / (2 * b4)) + c) / ((b1 / b2) + (b1 / b3) + (b1 / b4) + 1)
    l = a1 - 2 * b1 * x1
    x2 = (a2 - l) / (2 * b2)
    x3 = (a3 - l) / (2 * b3)
    x4 = (a4 - l) / (2 * b4)

    return [x1, x2, x3, x4]

if __name__ == '__main__':
    x_opt = real_solver()
    x_solver = solver_mini()
    plot_allocation(x_opt, x_solver)
