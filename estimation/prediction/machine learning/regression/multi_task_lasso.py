# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_multi_task_lasso_support.html

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import MultiTaskLasso, Lasso

rng = np.random.RandomState(42)

def data_create():
    n_obs, n_features, n_tasks = 100, 30, 40
    n_relevant_features = 5

    coef = np.zeros((n_tasks, n_features))
    times = np.linspace(0, 2 * np.pi, n_tasks)
    # print(times)
    # print(1 + rng.randn(1))
    # print(np.sin((1 + rng.randn(1))) * times)
    # sys.exit()
    for k in range(n_relevant_features):
        coef[:, k] = np.sin((1 + rng.randn(1)) * times + 3 * rng.randn(1))

    X = rng.randn(n_obs, n_features)
    Y = np.dot(X, coef.T) + rng.randn(n_obs, n_tasks)  # (100 x 30) * (30 x 40), which is the linear combination of features * coefficients

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(X[:, 0], Y[:, 0])
    plt.show()

def muti_task_lasso(df):
    coef_lasso_ = np.array([Lasso(alpha=0.5).fit(X, y).coef_ for y in Y.T])
    coef_multi_task_lasso_ = MultiTaskLasso(alpha=1.).fit(X, Y).coef_

def plot_predictions(df):
    fig = plt.figure(figsize=(8, 5))
    plt.subplot(1, 2, 1)
    plt.spy(coef_lasso_)
    plt.xlabel('Feature')
    plt.ylabel('Time (or Task)')
    plt.text(10, 5, 'Lasso')
    plt.subplot(1, 2, 2)
    plt.spy(coef_multi_task_lasso_)
    plt.xlabel('Feature')
    plt.ylabel('Time (or Task)')
    plt.text(10, 5, 'MultiTaskLasso')
    fig.suptitle('Coefficient non-zero location')

    feature_to_plot = 0
    plt.figure()
    lw = 2
    plt.plot(coef[:, feature_to_plot], color='seagreen', linewidth=lw,
             label='Ground truth')
    plt.plot(coef_lasso_[:, feature_to_plot], color='cornflowerblue', linewidth=lw,
             label='Lasso')
    plt.plot(coef_multi_task_lasso_[:, feature_to_plot], color='gold', linewidth=lw,
             label='MultiTaskLasso')
    plt.legend(loc='upper center')
    plt.axis('tight')
    plt.ylim([-1.1, 1.1])
    plt.show()

if __name__ == '__main__':
    data_create()