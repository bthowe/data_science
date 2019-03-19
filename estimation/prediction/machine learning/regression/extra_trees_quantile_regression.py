import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from skgarden import RandomForestQuantileRegressor, ExtraTreesQuantileRegressor
from sklearn.model_selection import train_test_split

def data_create():
    boston = load_boston()
    X = boston.data
    y = boston.target
    return train_test_split(X, y, test_size=.3, random_state=22)

def model_train(X_train, y_train):
    # rfqr = RandomForestQuantileRegressor(random_state=0, min_samples_split=10, n_estimators=100)
    rfqr = ExtraTreesQuantileRegressor(random_state=0, min_samples_split=10, n_estimators=100)
    rfqr.fit(X_train, y_train)
    return rfqr

def interval_predict(model, X_test, y_test):
    df = pd.concat(
        [
            pd.DataFrame(y_test, columns=['actual']),
            pd.DataFrame(model.predict(X_test, quantile=97.5), columns=['quantile_97.5']),
            pd.DataFrame(model.predict(X_test, quantile=50), columns=['median']),
            pd.DataFrame(model.predict(X_test, quantile=2.5), columns=['quantile_2.5'])
        ],
        axis=1
    )
    mean = (df['quantile_97.5'] + df['quantile_2.5']) / 2
    df['actual'] = df['actual'] - mean
    df['quantile_97.5'] = df['quantile_97.5'] - mean
    df['quantile_2.5'] = df['quantile_2.5'] - mean
    df['median'] = df['median'] - mean
    return df

def plotter(df, ordered=False):
    if ordered:
        df['diff'] = df['quantile_97.5'] - df['quantile_2.5']
        df.sort_values('diff', inplace=True)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(range(len(df)), df['actual'], color='red', label='Actual')
    ax.fill_between(range(len(df)), df['quantile_2.5'], df['quantile_97.5'], alpha=0.2, color='r')
    ax.scatter(range(len(df)), df['median'], color='blue', alpha=.4, label='Median')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = data_create()
    model = model_train(X_train, y_train)
    interval_predict(model, X_test, y_test).pipe(plotter, ordered=True)
