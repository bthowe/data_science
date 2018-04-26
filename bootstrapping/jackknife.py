import sys
import pprint
import numpy as np
import pandas as pd
from scipy.stats import binom
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def data_create():
    N = 1000

    beta0 = -1.6
    beta1 = -0.03
    beta2 = 0.6
    beta3 = 1.6

    df = pd.DataFrame(np.random.uniform(-1, 1, size=(N, 3)), columns=['X1', 'X2', 'X3'])
    df['pi_x'] = np.exp(beta0 + beta1 * df['X1'] + beta2 * df['X2'] + beta3 * df['X3']) / (1 + np.exp(beta0 + beta1 * df['X1'] + beta2 * df['X2'] + beta3 * df['X3']))
    df['target'] = binom.rvs(1, df['pi_x'])

    return df.drop('pi_x', 1)


def estimate_distribution(df, estimator):
    distribution = []
    for ind in range(len(df)):
        df_temp = df.copy()
        df_temp.drop(df_temp.index[ind], inplace=True)
        X = df_temp
        y = X.pop('target')

        estimator.fit(X, y)
        distribution.append(estimator.intercept_.tolist() + estimator.coef_.tolist()[0])
    return distribution

def plot_distribution(df):
    fig = plt.figure(figsize=(12, 8))
    for ind, column in enumerate(df):
        ax = fig.add_subplot(df.shape[1], 1, ind + 1)
        ax.hist(df[column], label=column)
        ax.legend()
    plt.show()

if __name__ == '__main__':
    x = data_create().pipe(estimate_distribution, LogisticRegression(C=100))
    df = pd.DataFrame(x, columns=['intercept', 'beta1', 'beta2', 'beta3'])
    plot_distribution(df)
