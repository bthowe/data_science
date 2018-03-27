import sys
import numpy as np
import pandas as pd
from scipy.stats import norm

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def _ci(mean, variance, alpha):
    """
    The true value of the parameter (the mean, in this case) is in the following interval 1 - alpha % of the time. Here,
    the randomness is from repeated sampling from the super population.
    """
    return mean - np.sqrt(variance) * norm.ppf(1 - alpha / 2), mean + np.sqrt(variance) * norm.ppf(1 - alpha / 2)

def T_strat(y, alpha):
    y_t = y.query('treatment == 1')
    y_c = y.query('treatment == 0')

    N_t = y_t['y'].groupby(y_t['stratum']).count()
    N_c = y_c['y'].groupby(y_c['stratum']).count()
    l = (N_t + N_c) / (N_t.sum() + N_c.sum())

    Y = y_t['y'].groupby(y_t['stratum']).mean() - y_c['y'].groupby(y_c['stratum']).mean()

    mean = (l * Y).sum()

    variance = ((l ** 2) * ((y_t['y'].groupby(y_t['stratum']).std() ** 2) / N_t) + ((y_c['y'].groupby(y_c['stratum']).std() ** 2) / N_c)).sum()

    return {"mean": mean, "variance": variance, "confidence interval": _ci(mean, variance, alpha)}


if __name__ == '__main__':
    np.random.seed(seed=2)
    N = 10
    J = 3
    df = pd.DataFrame(np.random.uniform(-1, 1, size=(N, 3)), columns=['one', 'two', 'three'])
    df['stratum'] = np.random.choice(range(J), size=(N, 1))
    df['treatment'] = np.random.choice([0, 1], size=(N, 1))
    df['y'] = np.random.uniform(0, 1, size=(N, 1))

    print(
        T_strat(
            df[['y', 'stratum', 'treatment']],
            0.05
        )
    )
