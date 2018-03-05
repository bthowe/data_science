import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def pi(X_c, X_t, covar, alpha):
    X_t[covar].quantile(alpha)
    stats.percentileofscore(X_c[covar], X_t[covar].quantile(alpha))

    pi_c = stats.percentileofscore(X_c[covar], X_t[covar].quantile(alpha)) / 100 + \
           (1 - stats.percentileofscore(X_c[covar], X_t[covar].quantile(1 - alpha)) / 100)
    pi_t = stats.percentileofscore(X_t[covar], X_c[covar].quantile(alpha)) / 100 + \
           (1 - stats.percentileofscore(X_t[covar], X_c[covar].quantile(1 - alpha)) / 100)
    return pi_c, pi_t

def multivariate_measure(X_c, X_t):
    """
    An estimate of the Mahalanobis distance between the means of data matrices X_c and X_t with respect to the inverse
    of the average of the variance/covariance matrices inner product.
    """
    # standardizing the data (i.e., using the same mean and variance to transform both datasets, does not change the measure by very much (one thousandth of a point)
    sigma_c = np.cov(X_c.values.T)
    sigma_t = np.cov(X_t.values.T)

    X_c_bar = X_c.mean().values
    X_t_bar = X_t.mean().values

    return np.sqrt(np.dot(np.dot((X_c_bar - X_t_bar), (np.linalg.pinv((sigma_c + sigma_t) / 2))), (X_c_bar - X_t_bar).T))


def mean_std(X_c, X_t):
    df_c = X_c.describe().loc[['mean', 'std']]
    df_t = X_t.describe().loc[['mean', 'std']]

    df = df_c.append(df_t).transpose()
    df.columns = ['mean_c', 'std_c', 'mean_t', 'std_t']

    df['nor_dif'] = (df['mean_c'] - df['mean_t']) / np.sqrt((df['std_c'] ** 2 + df['std_t'] ** 2) / 2)

    df['log_ratio_std'] = np.log(df['std_c']) - np.log(df['std_t'])

    return df

def overlap(df, X_c, X_t, alpha):
    df['pi_{}_test'.format(alpha)] = 0
    df['pi_{}_february'.format(alpha)] = 0

    for feature in df.index.tolist():
        pi_c, pi_t = pi(X_c, X_t, feature, alpha)
        df.loc[feature, 'pi_c_{}_test'.format(alpha)] = pi_c
        df.loc[feature, 'pi_t_{}_february'.format(alpha)] = pi_t

    return df

def hist_plotter(X1, X2, feature):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    if feature != 'target':
        ax.hist(X1[feature], label='Control', alpha=.5, normed=True)
        ax.hist(X2[feature], label='Treatment', alpha=.5, normed=True)
    else:
        ax.hist(X1, label='Control', alpha=.5, normed=True)
        ax.hist(X2, label='Treatment', alpha=.5, normed=True)
    plt.legend()
    plt.title('Histogram of {}'.format(feature))
    plt.savefig('/Users/travis.howe/Downloads/covariate_histograms_temp/{}.png'.format(feature))

def covariate_histograms(df, X_c, y_c, X_t, y_t):
    os.makedirs('/Users/travis.howe/Downloads/covariate_histograms_temp')
    for feature in df.index.tolist():
        hist_plotter(X_c, X_t, feature)
    hist_plotter(y_c, y_t, 'target')
    print(df)

def assess_covariate_distributions(X_c, y_c, X_t, y_t, alpha=0.05):
    print(mean_std(X_c, X_t).pipe(overlap, X_c, X_t, alpha)).pipe(covariate_histograms, X_c, y_c, X_t, y_t)
    print(multivariate_measure(X_c, X_t))

if __name__ == '__main__':
    N = 1000
    features = ['one', 'two', 'three', 'four', 'five']
    X1 = pd.DataFrame(np.random.uniform(-1, 1, size=(N, len(features))), columns=features)
    X2 = pd.DataFrame(np.random.uniform(-1, 1, size=(N, len(features))), columns=features)

    assess_covariate_distributions(X1, X2)


    # todo: how to intuitively interpret each measure
