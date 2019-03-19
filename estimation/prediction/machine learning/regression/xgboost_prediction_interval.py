"""
Qunatile regression implemented using a modification of the xgboost library. This is used to create point-wise
prediction intervals from the 1 - alpha and alpha quantiles.
See https://medium.com/bigdatarepublic/regression-prediction-intervals-with-xgboost-428e0a018b
"""
import sys
import numpy as np
import pandas as pd
from functools import partial
from scipy.stats import uniform
import matplotlib.pyplot as plt
from scipy.stats import binom_test
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split

class XGBOOSTQUANTILE(BaseEstimator, RegressorMixin):
    """XGBoost for quantile regression.

    Parameters
    ----------
    quant_alpha: The quantile to be estimated.

    quant_delta: Smoothing parameter

    quant_thres: The threshold

    quant_var: +/- Value of the gradient (each with probability 0.5) if x is beyond the absolute value of the threshold.
    """


    def __init__(self, quant_alpha, quant_delta, quant_thres, quant_var, n_estimators=100, max_depth=3, reg_alpha=5, reg_lambda=1, gamma=0.5):
        self.quant_alpha = quant_alpha
        self.quant_delta = quant_delta
        self.quant_thres = quant_thres
        self.quant_var = quant_var

        #xgboost parameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.gamma = gamma

        self.clf = None

    def _grad(self, x):
        if self.quant_thres < np.abs(x):
            return -(2 * np.random.randint(2) - 1) * self.quant_var
        elif x < (self.quant_alpha - 1) * self.quant_delta:
            return 1 - self.quant_alpha
        elif x < self.quant_alpha * self.quant_delta:  #(self.quant_alpha - 1) * self.quant_delta <= x < self.quant_alpha * self.quant_delta:
            return -x / self.quant_delta
        else:  # self.quant_alpha * self.quant_delta < x
            return -self.quant_alpha

    def _hess(self, x):
        if self.quant_thres <= np.abs(x):
            return 1
        elif x < (self.quant_alpha - 1) * self.quant_delta:
            return 0
        elif x < self.quant_alpha * self.quant_delta:
            return 1 / self.quant_delta
        else:
            return 0

    def _quantile_loss(self, y_true, y_pred):
        x = pd.Series(y_true - y_pred)
        return x.apply(self._grad), x.apply(self._hess)

    def fit(self, X, y):
        self.clf = XGBRegressor(
            objective=self._quantile_loss,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            gamma=self.gamma
        )
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        y_pred = self.clf.predict(X)
        return y_pred

    def score(self, X, y):
        y_pred = self.clf.predict(X)
        score = (self.quant_alpha-1.0)*(y-y_pred)*(y<y_pred)+self.quant_alpha*(y-y_pred)* (y>=y_pred)
        score = 1./np.sum(score)
        return score

def data_create():
    np.random.seed(1234513)
    n = 1000
    def f(x):
        return x * np.sin(x)

    X = np.atleast_2d(np.random.uniform(0, 10, size=n)).T
    X = X.astype(np.float32)
    y = f(X).ravel()
    dy = 1.5 + 1.0 * np.random.random(y.shape)
    noise = np.random.normal(0, dy)
    y += noise
    y = y.astype(np.float32)
    return pd.concat(
        [
            pd.DataFrame(X, columns=['X']),
            pd.DataFrame(y, columns=['y'])
        ],
        axis=1
    )

def data_create2():
    np.random.seed(1)
    n = 10000
    df = pd.DataFrame(np.atleast_2d(np.random.uniform(0, 10.0, size=n)).T, columns=['X'])
    df['y'] = df['X'] * np.sin(df['X']) + np.random.normal(0, 1, size=n)
    df.sort_values('X', inplace=True)
    return df

def _random_search(X, y, xgbq, xgb_parameters):
    grid_search = RandomizedSearchCV(
        xgbq,
        xgb_parameters,
        n_iter=100,
        scoring='neg_mean_absolute_error',
        verbose=10,
        n_jobs=-1,
        cv=3
    )
    grid_search.fit(X, y)
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    return grid_search

def model(df, alpha):
    X = df
    y = df.pop('y')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

    upper = _random_search(
        X_train, y_train,
        XGBOOSTQUANTILE(quant_alpha=1 - alpha / 2, quant_delta=1, quant_thres=6, quant_var=3.2),
        {'quant_delta': uniform(.01, 12), 'quant_thres': uniform(1, 12), 'quant_var': uniform(1, 12)}
    ).predict(X_test)

    lower = _random_search(
        X_train, y_train,
        XGBOOSTQUANTILE(quant_alpha=alpha / 2, quant_delta=1, quant_thres=6, quant_var=3.2),
        {'quant_delta': uniform(.01, 12), 'quant_thres': uniform(1, 12), 'quant_var': uniform(1, 12)}
    ).predict(X_test)

    median = _random_search(
        X_train, y_train,
        XGBOOSTQUANTILE(quant_alpha=.5, quant_delta=1, quant_thres=6, quant_var=3.2),
        {'quant_delta': uniform(.01, 12), 'quant_thres': uniform(1, 12), 'quant_var': uniform(1, 12)}
    ).predict(X_test)

    xgbls = XGBRegressor()
    xgbls.fit(X_train, y_train)
    mean = xgbls.predict(X_test)

    return pd.concat(
        [
            X_test.reset_index(drop=True),
            y_test.reset_index(drop=True),
            pd.DataFrame(upper, columns=['upper_bound']),
            pd.DataFrame(lower, columns=['lower_bound']),
            pd.DataFrame(mean, columns=['mean']),
            pd.DataFrame(median, columns=['median'])
        ],
        axis=1
    )

def plot(df, alpha):
    df.sort_values('X', inplace=True)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df['X'], df['mean'], color='seagreen', label='Mean')
    ax.plot(df['X'], df['median'], color='darksalmon', label='Median')
    ax.fill_between(df['X'], df['lower_bound'], df['upper_bound'], alpha=0.2, color='cornflowerblue', label='{}% Prediction Interval'.format((1 - alpha) * 100))
    ax.scatter(df['X'], df['y'], color='royalblue', alpha=0.3, label='Observations')
    ax.plot(df['X'], df['X'] * np.sin(df['X']), color='royalblue', label='$f(x) = x\, \sin(x)$', linestyle=':')
    ax.set_ylabel('$x\, \sin(x)$')
    ax.set_xlabel('$x$')
    plt.legend()
    plt.show()
    # plt.savefig('/Users/travis.howe/Downloads/temp.png')

if __name__ == '__main__':
    alpha = 0.1
    data_create().pipe(model, alpha).pipe(plot, alpha)
