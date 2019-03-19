import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.linear_model import LinearRegression

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)




def ls_coefficients(X, y):
    lr = LinearRegression(fit_intercept=True)
    lr.fit(X, y)

    coefficients = list(lr.coef_)
    coefficients.append(lr.intercept_)

    return dict(zip(X.columns.tolist(), coefficients))

def ls_treatment_effect_estimator(X, y, W):
    coeff_dic = ls_coefficients(X, y)
    return coeff_dic[W]

def V(df, W, Y, coeffs, type='hetero'):
    Y = df.pop(Y)
    W = df.pop(W)
    X = df.values

    N = len(df)
    M = X.shape[1]

    alpha = coeffs[0]
    tau = coeffs[1]
    betas = coeffs[2:]

    if type == 'hetero':
        return (1 / (N * (N - 1 - M))) * np.sum(((W - W.mean()) ** 2) * (Y - alpha - tau - betas * X) ** 2) * (1 / (W.mean() * (1 - W.mean())) ** 2)
    else:
        return (1 / (N * (N - 1 - M))) * np.sum((Y - alpha - tau - betas * X) ** 2) * (1 / (W.mean() * (1 - W.mean())))


def chi2_test(covars, results):
    V = results.cov_params()
    V_t_g = V[covars].loc[covars].values

    params = results.params.loc[covars].values
    return 1 - chi2.cdf(params.T.dot(V_t_g).dot(params), len(params))

def features_create(df):
    df['stratum_Nj'] = df['y'].groupby(df['stratum']).transform('count')

    strata_lst = df['stratum'].unique()
    strata_lst.sort()
    J = strata_lst[-1]
    N_J = len(df.query('stratum == {}'.format(J)))

    features_keep = []
    df['B_{}'.format(J)] = 0
    df.loc[df['stratum'] == J, 'B_{}'.format(J)] = 1
    features_keep.append('B_{}'.format(J))
    df['W_B_{}'.format(J)] = df['treatment'] * df['B_{}'.format(J)] / (N_J / len(df))
    features_keep.append('W_B_{}'.format(J))
    for stratum in strata_lst[:-1]:
        N_j = len(df.query('stratum == {}'.format(J)))
        df['B_{}'.format(stratum)] = 0
        df.loc[df['stratum'] == stratum, 'B_{}'.format(stratum)] = 1
        features_keep.append('B_{}'.format(stratum))
        df['W_B_{}'.format(stratum)] = df['treatment'] * (
        df['B_{}'.format(stratum)] - df['B_{}'.format(J)] * (N_j / N_J))
        features_keep.append('W_B_{}'.format(stratum))
    return df[features_keep + ['y']]

def T_old_inter(df):
    # Avoid multicollinearity by not including an intercept.
    df = features_create(df)

    X = df
    y = X.pop('y')

    model = sm.OLS(y, X)
    results = model.fit()
    print(results.params)

# todo: is this correct?
# todo: variance of tau
# todo: first specification from book



if __name__ == '__main__':
    np.random.seed(seed=2)
    N = 10
    J = 3
    df = pd.DataFrame(np.random.uniform(-1, 1, size=(N, 3)), columns=['one', 'two', 'three'])
    df['stratum'] = np.random.choice(range(J), size=(N, 1))
    df['treatment'] = np.random.choice([0, 1], size=(N, 1))
    df['y'] = np.random.uniform(0, 1, size=(N, 1))

    T_old_inter(df)
