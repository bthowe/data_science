import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom, uniform, expon

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
from skopt.space import Real, Integer, Categorical
from sklearn.model_selection import cross_val_score

np.random.seed(42)

def data_create():
    n = 10000

    b0 = -1.6
    b1 = -0.03
    b2 = 0.6
    b3 = 1.6

    df = pd.DataFrame(np.random.uniform(-1, 1, size=(n, 3)), columns=['X1', 'X2', 'X3'])
    df['pi_x'] = np.exp(b0 + b1 * df['X1'] + b2 * df['X2'] + b3 * df['X3']) / (1 + np.exp(b0 + b1 * df['X1'] + b2 * df['X2'] + b3 * df['X3']))

    df_sample = df.sample(frac=.1)
    df.loc[df_sample.index, 'pi_x'] = uniform.rvs(0, 1, size=(len(df_sample),))

    df['y'] = binom.rvs(1, df['pi_x'])

    return df.drop('pi_x', 1)

def random_search(df):
    X = df
    y = X.pop('y')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

    lr_parameters = {
        'penalty': ['l1', 'l2'],
        'C': expon,
        'fit_intercept': [True, False],
        'class_weight': [None, 'balanced']
    }

    grid_search = RandomizedSearchCV(
        LogisticRegression(),
        lr_parameters,
        n_iter=200,
        scoring='roc_auc',
        # scoring='neg_log_loss',
        verbose=0,
        n_jobs=1,
        cv=5
    )
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    print(grid_search.best_estimator_.coef_)


def skopt_search(df):
    X = df
    y = X.pop('y')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

    lr_param_space = [
        Categorical(['l1', 'l2'], name='penalty'),
        Real(10 ** -5, 5, "log-uniform", name='C'),
        Categorical([True, False], name='fit_intercept'),
        Categorical([None, 'balanced'], name='class_weight')
    ]

    clsfr = LogisticRegression()

    @use_named_args(lr_param_space)
    def objective(**params):
        clsfr.set_params(**params)

        return -np.mean(cross_val_score(clsfr, X_train, y_train, verbose=0, cv=5, n_jobs=-1, scoring="roc_auc"))

    res_gp = gp_minimize(objective, lr_param_space, n_calls=50, random_state=0)

    print('Best score: {}'.format(res_gp.fun))
    print('''Best parameters:\n
    \t- penalty={0}\n
    \t- C={1}\n
    \t- fit_intercept={2}\n
    \t- class_weight={3}
    '''.format(res_gp.x[0], res_gp.x[1], res_gp.x[2], res_gp.x[3]))
    # plot_convergence(res_gp)
    # plt.show()

    # todo: time each of these.
    # todo: what are the coefficients here?
    # todo: how do you save the estimator?

if __name__ == '__main__':
    df = data_create()
    random_search(df.copy())
    skopt_search(df.copy())
