import sys
import itertools
import statistics
import numpy as np
import pandas as pd

def fishers_exact_test(df, statistic, alpha=0.05):
    """
    Performs Fishers exact test ...

    :param df: original dataframe
    :param statistic: statistic used to evaluate treatment effect
    :param alpha: determines the 1-alpha% "Fisher" interval, i.e., the p-values larger than alpha

    :return: the value of the test-statistic, the exact p-value of the test statistic, and the 1-alpha% "Fisher" interval for the treatment effect.
    """
    # todo: could always just take M sample permutations, make as option

    test_stat = statistic(df, 'outcome', 'W')

    t_indeces = [list(t_indeces) for t_indeces in itertools.combinations(range(len(df)), df['W'].sum())]
    t_indeces.remove(df.query('W == 1').index.tolist())
    stat_distribution = []
    for i in t_indeces:
        df['W'] = 0
        df.loc[i, 'W'] = 1
        stat_distribution.append(statistic(df, 'outcome', 'W'))

    return test_stat, \
           np.sum(np.where(stat_distribution >= test_stat, 1, 0)) / len(stat_distribution), \
           (np.percentile(stat_distribution, alpha*100), np.percentile(stat_distribution, (1-alpha) * 100))


if __name__ == '__main__':
    N = 10
    np.random.seed(2)
    columns = ['outcome']
    df = pd.DataFrame(np.random.uniform(-1, 1, size=(N, len(columns))), columns=columns)
    df['W'] = np.random.randint(0, 2, size=(N, 1))

    stat = statistics.abs_ave_by_treatment

    print(fishers_exact_test(df, stat, alpha=.05))
