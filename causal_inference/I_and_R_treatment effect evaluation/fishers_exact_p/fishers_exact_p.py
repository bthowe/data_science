import itertools

import numpy as np
import pandas as pd

from fishers_exact_p import fep_stats


def fishers_exact_test(df, statistic, alpha=0.05):
    """
    Performs Fishers exact test on the data contained in df.

    :param df: original dataframe
    :param statistic: statistic used to evaluate treatment effect
    :param alpha: determines the 1-alpha% "Fisher" interval, i.e., the p-values larger than alpha

    :return: the value of the test-statistic, the exact p-value of the test statistic, and the 1-alpha% "Fisher" interval for the treatment effect.
    """
    test_stat = statistic(df, 'outcome', 'W')

    t_indeces = [list(t_indeces) for t_indeces in itertools.combinations(range(len(df)), df['W'].sum())]
    t_indeces.remove(df.query('W == 1').index.tolist())
    stat_distribution = []
    df_draw = df.copy()
    for i in t_indeces:
        df_draw['W'] = 0
        df_draw.loc[i, 'W'] = 1
        stat_distribution.append(statistic(df_draw, 'outcome', 'W'))

    return test_stat, \
           np.sum(np.where(stat_distribution >= test_stat, 1, 0)) / len(stat_distribution), \
           (np.percentile(stat_distribution, alpha*100), np.percentile(stat_distribution, (1-alpha) * 100))

def fishers_exact_test_approx(df, statistic, alpha=0.05, samples=1000, replacement=True):
    """
    Approximates Fishers exact test on the data contained in df by taking random draws with replacement from the set of
    all random permutations of the vector of treatment indicators.


    :param df: original dataframe
    :param statistic: statistic used to evaluate treatment effect
    :param alpha: determines the 1-alpha% "Fisher" interval, i.e., the p-values larger than alpha

    :return: the value of the test-statistic, the exact p-value of the test statistic, and the 1-alpha% "Fisher" interval for the treatment effect.
    """
    test_stat = statistic(df, 'outcome', 'W')

    stat_distribution = []
    df_draw = df.copy()
    for draw in range(samples):
        df_draw['W'] = df['W'].sample(frac=1, replace=False).reset_index(drop=True)
        stat_distribution.append(statistic(df_draw, 'outcome', 'W'))

    return test_stat, \
           np.sum(np.where(stat_distribution >= test_stat, 1, 0)) / len(stat_distribution), \
           (np.percentile(stat_distribution, alpha * 100), np.percentile(stat_distribution, (1 - alpha) * 100))


if __name__ == '__main__':
    N = 10
    # np.random.seed(2)
    columns = ['outcome']
    df = pd.DataFrame(np.random.uniform(-1, 1, size=(N, len(columns))), columns=columns)
    df['W'] = np.random.randint(0, 2, size=(N, 1))

    stat = fep_stats.abs_ave_by_treatment

    print(fishers_exact_test(df, stat, alpha=.05))
    print(fishers_exact_test_approx(df, stat, alpha=.05))
