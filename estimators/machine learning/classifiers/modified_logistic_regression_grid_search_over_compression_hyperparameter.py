import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, binom
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)  # per, https://github.com/statsmodels/statsmodels/issues/3931

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def data_create():
    np.random.seed(1)

    n = 1000
    m = 1
    b0 = 0
    b1 = 0.5

    df = pd.DataFrame(np.random.uniform(-5, 5, size=(n, m)), columns=['x1'])
    df['x0'] = 1
    df = df[['x0', 'x1']]
    df['prob'] = 1 / (1 + np.exp(-(b0 + df['x1'] * b1)))
    df['y'] = binom.rvs(1, df['prob'])
    return df.drop('prob', 1)

def plot_ci(model, alpha):
    """
    Plots the predicted probabilities as well as a (1 - alpha) * 100 % confidence interval.

    :param model: fitted model, an sklearn RandomizedSearchCV or GridSearchCV object
    :param alpha: level of significance, used to calculate the confidence interval
    """

    X = np.c_[np.ones(shape=(100, 1)), np.linspace(-10, 10, 100)]
    proba = model.predict_proba(pd.DataFrame(X))[:, 1]

    a = model.best_params_['a']
    cov = model.best_estimator_.model.cov_params()

    gradient = (proba * (a - proba) * X.T).T  # matrix of gradients for each observation
    std_errors = np.array([np.sqrt(np.dot(np.dot(g, cov), g)) for g in gradient])

    c = norm.ppf(1 - (alpha / 2))

    upper = np.maximum(0, np.minimum(1, proba + std_errors * c))
    lower = np.maximum(0, np.minimum(1, proba - std_errors * c))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(X[:, 1], proba, label='predicted probability', color='b')
    ax.plot(X[:, 1], lower, label='lower 95% CI', color='g')
    ax.plot(X[:, 1], upper, label='upper 95% CI', color='g')
    ax.set_ylabel('Probability')
    ax.set_xlabel('x1 value')
    plt.legend()
    plt.show()

class LogRegWrapper(BaseEstimator, ClassifierMixin):
    """
    This class modifies statsmodel's Logit class by multiplying the predicted probabilities by a scalar between 0 and 1.
    This scalar functions as a hyperparameter, and governs how squished the sigmoid is.
    """
    def __init__(self, a=1, threshold=0.5):
        self.a = a
        self.threshold = threshold

    def fit(self, X, y=None):
        self.model = sm.Logit(y, X).fit()
        return self

    def predict_proba(self, X, y=None):
        prob = (self.model.predict(X) * self.a).values
        return np.c_[1 - prob, prob]

    def predict(self, X, y=None):
        return np.where(self.model.predict(X) * self.a > self.threshold, 1, 0)

def model_train(X, y):
    lrw = LogRegWrapper()
    param_grid = {'a': uniform(.9, .1)}
    grid_search = RandomizedSearchCV(lrw, param_grid, n_iter=10, scoring='roc_auc', verbose=10, n_jobs=-1, cv=5)
    grid_search.fit(X, y)
    print(grid_search)
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    return grid_search

if __name__ == '__main__':
    df = data_create()
    X = df
    y = X.pop('y')

    model = model_train(X, y)
    plot_ci(model, 0.05)
