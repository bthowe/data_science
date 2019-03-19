import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import uniform
from scipy.stats import norm
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import MultiTaskLasso, Lasso
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def data_create():
    np.random.seed(seed=2)

    n_obs = 1000
    n_outcomes = 3
    n_features = 1

    beta = np.array([-1.6, -0.03])
    # beta = np.array([-1.6, -0.03, 0.6, 1.6])

    coefs = np.zeros((n_features + 1, n_outcomes))
    for coef in range(n_outcomes):
        coefs[:, coef] = beta + norm.rvs(0, 1, size=(n_features + 1, ))

    df = pd.concat(
        [
            pd.DataFrame(np.ones((n_obs, 1)), columns=['X0']),
            pd.DataFrame(np.random.uniform(-1, 1, size=(n_obs, n_features)), columns=['X1'])
            # pd.DataFrame(np.random.uniform(-1, 1, size=(n_obs, n_features)), columns=['X1', 'X2', 'X3'])
        ],
        axis=1
    )
    X = df.values
    df[['y1', 'y2', 'y3']] = pd.DataFrame(np.dot(X, coefs)) + norm.rvs(0, 1, size=(n_obs, n_outcomes))
    return df

def mtl_roc_auc(y_true, y_pred):
    return mean_absolute_error(y_true.values.flatten(), y_pred.flatten())

def multi_task_lasso(df):
    X = df[['X0', 'X1']]
    # X = df[['X0', 'X1', 'X2', 'X3']]
    Y = df[['y1', 'y2', 'y3']]

    mtl_scorer = make_scorer(mtl_roc_auc, greater_is_better=True)
    mtl_parameters = {
        'alpha': uniform(0, 10)
    }

    grid_search = RandomizedSearchCV(
        MultiTaskLasso(fit_intercept=False, alpha=0.05),
        mtl_parameters,
        n_iter=200,
        scoring=mtl_scorer,
        verbose=10,
        n_jobs=1,
        cv=5
    )
    grid_search.fit(X, Y)
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    print(grid_search.best_estimator_.coef_)


def plotter(df):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(3, 1, 1)
    ax.scatter(df['X1'], df['y1'], color='seagreen', alpha=.2)
    ax.plot(df['X1'], df['y1_predict'], color='seagreen')
    ax = fig.add_subplot(3, 1, 2)
    ax.scatter(df['X1'], df['y2'], color='cornflowerblue', alpha=.2)
    ax.plot(df['X1'], df['y2_predict'], color='cornflowerblue')
    ax = fig.add_subplot(3, 1, 3)
    ax.scatter(df['X1'], df['y3'], color='darksalmon', alpha=.2)
    ax.plot(df['X1'], df['y3_predict'], color='darksalmon')
    plt.show()


def lasso(df):
    X = df[['X0', 'X1']]
    # X = df[['X0', 'X1', 'X2', 'X3']]
    Y = df[['y1', 'y2', 'y3']]

    from scipy.stats import expon
    l_parameters_rs = {
        'alpha': expon()
    }
    # l_parameters_rs = {
    #     'alpha': uniform(0, 10)
    # }

    l_parameters_gs = {
        'alpha': [0.01, 0.1, 0, 1, 10]
    }

    for y in Y:
        # l = Lasso(fit_intercept=False, alpha=0.0)
        # l.fit(X, Y[y])
        # df['{}_predict'.format(y)] = l.predict(X)
        # print(l.coef_)

        grid_search = RandomizedSearchCV(
            Lasso(fit_intercept=False, alpha=0.0),
            l_parameters_rs,
            n_iter=200,
            scoring='neg_mean_squared_error',
            verbose=0,
            n_jobs=-1,
            cv=5
        )
        grid_search.fit(X, Y[y])
        print(grid_search.best_params_)
        print(grid_search.best_score_)
        df['{}_predict'.format(y)] = grid_search.predict(X)
        print(grid_search.best_estimator_.coef_)
        #
        # grid_search = GridSearchCV(
        #     Lasso(fit_intercept=False, alpha=0.0),
        #     l_parameters_gs,
        #     scoring='neg_mean_squared_error',
        #     verbose=0,
        #     n_jobs=-1,
        #     cv=5
        # )
        # grid_search.fit(X, Y[y])
        # print(grid_search.best_params_)
        # print(grid_search.best_score_)
        # df['{}_predict'.format(y)] = grid_search.predict(X)
        # print(grid_search.best_estimator_.coef_)

    # print(np.array([Lasso(fit_intercept=False, alpha=0.0).fit(X, Y[y]).coef_ for y in Y]))

    return df
    # print(MultiTaskLasso(fit_intercept=False, alpha=0.05).fit(X, Y).coef_)

if __name__ == '__main__':
    # df = data_create().pipe(plotter)
    # df = data_create().pipe(multi_task_lasso)
    data_create().pipe(lasso).pipe(plotter)

# todo: gridsearch
# todo: compare fits and coefficients
# todo: maybe different data would be better.

