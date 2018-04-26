import numpy as np
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt

def data_create():
    np.random.seed(1)
    x = np.arange(100)
    y = (x * 0.5 + np.random.normal(size=100, scale=10) > 30)
    return x, y

def log_reg(x, y):
    X = sm.add_constant(x)
    model = sm.Logit(y, X).fit()
    return model

def plot_ci(model, x, alpha):
    """
    Plots the predicted probabilities as well as a (1 - alpha) * 100 % confidence interval.

    :param model: fitted statsmodel logistic regression model
    :param x: the matrix of covariates without a constant
    :param alpha: level of significance
    """
    X = sm.add_constant(x)

    proba = model.predict(X)
    cov = model.cov_params()

    gradient = (proba * (1 - proba) * X.T).T  # matrix of gradients for each observation
    std_errors = np.array([np.sqrt(np.dot(np.dot(g, cov), g)) for g in gradient])

    c = norm.ppf(1 - (alpha / 2))

    upper = np.maximum(0, np.minimum(1, proba + std_errors * c))
    lower = np.maximum(0, np.minimum(1, proba - std_errors * c))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, proba, label='predicted probability', color='b')
    ax.plot(x, lower, label='lower 95% CI', color='g')
    ax.plot(x, upper, label='upper 95% CI', color='g')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    x, y = data_create()
    model = log_reg(x, y)
    plot_ci(model, x, .05)