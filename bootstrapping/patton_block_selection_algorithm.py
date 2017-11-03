import sys
import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.stattools import acf, acovf, ccf

pd.set_option('max_columns', 700)
pd.set_option('max_info_columns', 100000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

def lam(x):
    return np.where((np.abs(x) >= 0), 1, 0) * np.where((np.abs(x) < 0.5), 1, 0) + 2 * (1 - np.abs(x)) * np.where((np.abs(x) >= 0.5), 1, 0) * np.where((np.abs(x) <= 1), 1, 0)

def find_M(data, mmax, Kn, rho_k_crit):
    rho_k = acf(data, nlags=mmax)[1:]

    ni_function = lambda x: np.sum((np.abs(rho_k) < rho_k_crit)[x: x + Kn])
    num_insignificant = [ni_function(i) for i in xrange(mmax - Kn + 1)]

    if Kn in num_insignificant:
        mhat = num_insignificant.index(Kn)
    else:
        if any(np.abs(rho_k) > rho_k_crit):
            lag_sig = np.where(rho_k > rho_k_crit)[0]
            if len(lag_sig) == 1:
                mhat = lag_sig[0]
            else:
                mhat = np.max(lag_sig)
        else:
            mhat = 1

    if 2 * (mhat + 1) > mmax:
        M = mmax
    else:
        M = 2 * (mhat + 1)

    return M

def find_G_D(data, bootstrap_type, M):
    kk = np.arange(-M, M + 1)

    acov = acovf(data)[: M + 1]
    R_k = np.r_[acov[1:][::-1], acov]

    Ghat = np.sum(lam(kk / float(M)) * np.abs(kk) * R_k)

    if bootstrap_type == 'Circular':
        Dhat = (4 / 3.) * np.sum(lam(kk / float(M)) * R_k) ** 2
    else:
        Dhat = 2 * np.sum(lam(kk / float(M)) * R_k) ** 2

    return Ghat, Dhat

def rounder(x, bootstrap_type):
    if x < 1:
        return 1
    else:
        if bootstrap_type == 'Circular':
            return np.round(x)
        else:
            return np.ceil(x)

def find_bstar(bootstrap_type, rnd, Ghat, Dhat, Bmax, n):
    if not rnd:
        bstar = min(((2 * Ghat ** 2) / Dhat) ** (1 / 3.) * n ** (1 / 3.), Bmax)
    else:
        bstar = min(rounder(((2 * Ghat ** 2) / Dhat) ** (1 / 3.) * n ** (1 / 3.), bootstrap_type), Bmax)
    return bstar

def opt_block_length(df, bootstrap_type='Circular', rnd=False):
    """This is a function taken from Andrew Patton (http://public.econ.duke.edu/~ap172/) to select the optimal (in the
    sense of minimising the MSE of the estimator of the long-run variance) block length for the stationary bootstrap or
    circular bootstrap. Code follows Politis and White, 2001, 'Automatic Block-Length Selection for the Dependent
    Bootstrap.'

    INPUTS:
    -data, an nxk pandas dataframe
    -bootstrap_type, a string indicating whether the optimal bootstrap block lengths should be found for the stationary
    bootstrap or the circular bootstrap
    -rnd, whether the estimtes should be rounded or not

    OUTPUTS: Bstar, a 2xk vector of optimal bootstrap block lengths, [BstarSB;BstarCB]

"""
    n, k = df.shape

    c = norm.ppf(0.975)
    Kn = int(np.maximum(5, np.sqrt(np.log10(n))))

    mmax = int(np.ceil(np.sqrt(n)) + Kn)
    Bmax = np.ceil(np.minimum(3 * np.sqrt(n), n / 3))

    rho_k_crit = c * np.sqrt(np.log10(n) / n)

    Bstar = []

    for i in xrange(0, k):
        data = df.iloc[:, i]

        M = find_M(data, mmax, Kn, rho_k_crit)
        Ghat,  Dhat = find_G_D(data, bootstrap_type, M)
        bstar = find_bstar(bootstrap_type, rnd, Ghat, Dhat, Bmax, n)

        Bstar.append(bstar)
    return Bstar









if __name__ == '__main__':
    # df = pd.DataFrame([[0.3004800, 0.4806089, -1.0232758, 0.9733925, 0.2688148, -0.5161701, -0.7270535, -1.7454589,
    #                     -0.1558424, -0.6287940],
    #                    [0.3814848, 0.3526421, 0.8240130, 0.4161907, -0.6108061, -2.3804401, 1.4487400, 0.7112594,
    #                     -0.4536392, 0.8210167]]).transpose()

    df = pd.read_csv('/Users/travis.howe/Downloads/patton_x.csv', index_col=False).drop('Unnamed: 0', 1)
    print opt_block_length(df, bootstrap_type='Stationary', rnd=False)
