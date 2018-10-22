import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
from scipy.stats import binom, norm
from sklearn.neighbors import DistanceMetric
from sklearn.linear_model import LogisticRegression

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)



def propensity_score(df):
    X = df
    y = df.pop('treatment')

    r = LogisticRegression(penalty='l2')
    r.fit(X, y)
    return r.predict_proba(X)[:, 1]

def data_create():
    N = 10
    effect = .1

    b0 = -1.6
    b1 = -0.03
    b2 = 0.6
    b3 = 1.6

    df = pd.DataFrame(np.random.uniform(-1, 1, size=(N, 3)), columns=['one', 'two', 'three'])
    df['treatment'] = np.random.randint(0, 2, size=(len(df), 1))

    def sigmoid(x):
        return np.exp(x) / (1 + np.exp(x))

    def treatment_effect(x):
        if x['treatment'] == 0:
            return sigmoid(b0 + x['one'] * b1 + x['two'] * b2 + x['three'] * b3)
        else:
            return sigmoid(b0 + x['one'] * b1 + x['two'] * b2 + x['three'] * b3) + effect
    df['p'] = df.apply(treatment_effect, axis=1)
    df['y'] = binom.rvs(1, df['p'])

    print('Estimate of treatment effect: {}'.format(df.query('treatment == 1')['p'].mean() - df.query('treatment == 0')['p'].mean()))

    nt = df.query('treatment == 1').shape[0]
    nc = df.query('treatment == 0').shape[0]
    p = df['y'].mean()

    print('Estimate of standard error: {}'.format(np.sqrt(p * (1-p) * ((1 / nt) + (1 / nc)))))
    print('\n\n')
    print('Difference in means: {}'.format(df.query('treatment == 1')['y'].mean() - df.query('treatment == 0')['y'].mean()))

    df['constant'] = 1

    df['propensity_score'] = propensity_score(df.copy()[['treatment', 'one', 'two', 'three']])


    return df[['y', 'treatment', 'propensity_score', 'one', 'two', 'three']]



def weight_matrix(X_t, X_c, type):
    V = (np.cov(X_t.T) + np.cov(X_c.T)) / 2
    if type == 'euclidean':
        return np.diag(V)
    return V


def inexact_matching_without_replacement(X, covars, propensity):
    from sklearn.neighbors import DistanceMetric

    X['p'] = propensity
    X = X.sort_values('p', ascending=False).reset_index(drop=True)

    X_t = X.query('treatment == 1')[covars]
    X_c = X.query('treatment == 0')[covars]

    V = weight_matrix(X_t[covars], X_c[covars], 'mahalanobis')

    dist = DistanceMetric.\
        get_metric('mahalanobis', V=V).\
        pairwise(X[covars])


    # print(X)
    t_ind = list(X_t.index)
    c_ind = list(X_c.index)
    print(t_ind)
    print(c_ind)
    dist2 = dist[t_ind, :][:, c_ind]
    print(dist2)

    print(np.argmin(dist2, axis=1))



    lst = []
    for t_i in t_ind:
        m = np.argmin(dist[[t_i], :][:, c_ind], axis=1)[0]
        lst.append(c_ind[m])
        c_ind.remove(c_ind[m])
    print(lst)

    X_t['matches'] = lst

    print(X_t)


    # todo: check to make sure this is correct
    # todo: what should the output be?
    sys.exit()

    from numpy.linalg import inv

    xt = X_t.iloc[0:1].values
    xc = X_c.iloc[1:2].values

    print(np.sqrt((xt - xc) @ inv(V) @ (xt - xc).T))


    sys.exit()



    from sklearn.neighbors import DistanceMetric

    dist = DistanceMetric.get_metric('mahalanobis', V=np.cov(X_t))
    dist.pairwise(X_t)


    # would be better to score the entire df in at once.
    match_indeces = []
    for ind_t, row_t in X_t:
        d = distance(row_t, X_c)
        np.argmin(d)





if __name__ == '__main__':
    df = data_create()
    covars = ['one', 'two', 'three']
    inexact_matching_without_replacement(df[['treatment'] + covars], covars, df['propensity_score'])


# todo: come up with fake data for which I know the treatment effect
# todo: implement everything in python
# todo: implement everything in R
