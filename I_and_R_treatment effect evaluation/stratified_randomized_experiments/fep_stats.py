import sys
import numpy as np
import pandas as pd

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def T_diff_lambda(y_t, y_c, lam='RSS'):
    Y_t = y_t['y'].groupby(y_t['stratum']).mean()
    N_t = y_t['y'].groupby(y_t['stratum']).count()
    Y_c = y_c['y'].groupby(y_c['stratum']).mean()
    N_c = y_c['y'].groupby(y_c['stratum']).count()

    if lam == 'RSS':
        l = (N_t + N_c) / (N_t.sum() + N_c.sum())
        return (l * (Y_t - Y_c)).sum()
    elif lam == 'OPT':
        l = ((N_t + N_c) * (N_t / (N_t + N_c)) * (N_c / (N_t + N_c))) / ((N_t + N_c) * (N_t / (N_t + N_c)) * (N_c / (N_t + N_c))).sum()
        return (l * (Y_t - Y_c)).sum()

def T_rank_stratum(y):
    y['rank'] = y['y'].groupby(y['stratum']).rank()
    y['norm'] = (y['y'].groupby(y['stratum']).transform('count') + 1) / 2
    y['normalized_rank'] = y['rank'] - y['norm']
    return np.abs(y.query('treatment == 1')['normalized_rank'].mean() - y.query('treatment == 0')['normalized_rank'].mean())

def T_range(y):
    y_t = y.query('treatment == 1')
    y_c = y.query('treatment == 0')

    Y_t = y_t['y'].groupby(y_t['stratum']).max() - y_t['y'].groupby(y_t['stratum']).min()
    Y_c = y_c['y'].groupby(y_c['stratum']).max() - y_c['y'].groupby(y_c['stratum']).min()

    N_t = y_t['y'].groupby(y_t['stratum']).count()
    N_c = y_c['y'].groupby(y_c['stratum']).count()
    l = (N_t + N_c) / (N_t.sum() + N_c.sum())

    return (l * (Y_t - Y_c)).sum()

if __name__ == '__main__':
    np.random.seed(seed=2)
    N = 10
    J = 3
    df = pd.DataFrame(np.random.uniform(-1, 1, size=(N, 3)), columns=['one', 'two', 'three'])
    df['stratum'] = np.random.choice(range(J), size=(N, 1))
    df['treatment'] = np.random.choice([0, 1], size=(N, 1))
    df['y'] = np.random.uniform(0, 1, size=(N, 1))

    print(
        T_diff_lambda(
            df.query('treatment == 1')[['y', 'stratum']],
            df.query('treatment == 0')[['y', 'stratum']],
            lam='OPT'
            # lam='RSS'
        )
    )

    print(
        T_rank_stratum(
            df[['y', 'stratum', 'treatment']]
        )
    )

    print(
        T_range(
            df[['y', 'stratum', 'treatment']]
        )
    )
