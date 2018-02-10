import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.stats import norm

def neyman_estimator(df, W, Y):
    """
    Returns the Neyman estimator for the average treatment effect, which is the difference between the average
    outcomes for the treatment and the control groups.

    df: the dataset
    W: the treatment assignment variable
    Y: the outcome
    """
    return df.loc[df[W] == 1][Y].mean() - df.loc[df[W] == 0][Y].mean()

def V_neyman(df, W, Y):
    """
    A conservative estimate of the sampling variance but allows for unrestricted treatment effect heterogeneity.
    """
    df_t = df.loc[df[W] == 1][Y]
    n_t = len(df_t)

    df_c = df.loc[df[W] == 0][Y]
    n_c = len(df_c)

    s2_t = ((df_t - df_t.mean()) ** 2).sum() / (n_t - 1)
    s2_c = ((df_c - df_c.mean()) ** 2).sum() / (n_c - 1)

    return s2_t / n_t + s2_c / n_c

def V_rho(df, W, Y):
    """
    Another conservative estimate of the sampling variance but exploits differences in the variances of the outcome
    by treatment group. Also allows for unrestricted treatment effect heterogeneity.
    """
    n = len(df)

    df_t = df.loc[df[W] == 1][Y]
    n_t = len(df_t)

    df_c = df.loc[df[W] == 0][Y]
    n_c = len(df_c)

    s2_t = ((df_t - df_t.mean()) ** 2).sum() / (n_t - 1)
    s2_c = ((df_c - df_c.mean()) ** 2).sum() / (n_c - 1)

    return s2_c * (n_t / (n * n_c)) + s2_t * (n_c / (n * n_t)) + np.sqrt(s2_c) * np.sqrt(s2_t) * (2 / N)

def V_const(df, W, Y):
    """
    This estimate of the sampling variance relies on a constant treatment effect assumption to be valid.
    """
    n = len(df)

    df_t = df.loc[df[W] == 1][Y]
    n_t = len(df_t)

    df_c = df.loc[df[W] == 0][Y]
    n_c = len(df_c)

    s2 = (((df_c - df_c.mean()) ** 2).sum() + ((df_t - df_t.mean()) ** 2).sum()) / (n - 2)
    return s2 * (1 / n_c + 1 / n_t)

def confidence_interval(t_diff, V2, alpha=.1, randomization_distr='normal'):
    if randomization_distr == 'normal':
        return (t_diff - np.sqrt(V2) * norm.ppf(1 - alpha / 2), t_diff + np.sqrt(V2) * norm.ppf(1 - alpha / 2))
    # elif randomization_distr == 't':
    #     return (t_diff - np.sqrt(V2) * t.ppf(1 - alpha / 2, df=n), t_diff + np.sqrt(V2) * t.ppf(1 - alpha / 2, df=n))




if __name__ == '__main__':
    N = 10
    # np.random.seed(2)
    columns = ['outcome']
    df = pd.DataFrame(np.random.uniform(-1, 1, size=(N, len(columns))), columns=columns)
    df['W'] = np.random.randint(0, 2, size=(N, 1))

    print(neyman_estimator(df, 'W', 'outcome'))
    print(V_neyman(df, 'W', 'outcome'))
    print(V_rho(df, 'W', 'outcome'))
    print(V_const(df, 'W', 'outcome'))
