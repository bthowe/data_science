"""
Using sklearn's GradientBoostingRegressor library to create point-wise prediction intervals using quantile regression.
See http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor

np.random.seed(1)


def data_create():
    n = 100
    df = pd.DataFrame(np.atleast_2d(np.random.uniform(0, 10.0, size=n)).T, columns=['X'])
    df['y'] = df['X'] * np.sin(df['X']) + np.random.normal(0, 1, size=n)
    df.sort_values('X', inplace=True)
    return df

def model(df, alpha):
    X = df
    y = df.pop('y')

    # as expected, {'loss': 'quantile', 'alpha': .5} is the same as {'loss': 'lad'}
    clf= GradientBoostingRegressor(
        loss='quantile',
        alpha=1 - alpha/2,
        n_estimators=250,
        max_depth=3,
        learning_rate=.1,
        min_samples_leaf=9,
        min_samples_split=9
    )
    clf.fit(X, y)
    upper = clf.predict(X)

    clf.set_params(alpha=alpha / 2)
    clf.fit(X, y)
    lower = clf.predict(X)

    clf.set_params(loss='ls')
    clf.fit(X, y)
    mean = clf.predict(X)

    clf.set_params(loss='lad')
    clf.fit(X, y)
    median = clf.predict(X)

    return pd.concat(
        [
            X.reset_index(drop=True),
            y.reset_index(drop=True),
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


if __name__ == '__main__':
    alpha = 0.05
    data_create().pipe(model, alpha).pipe(plot, alpha)