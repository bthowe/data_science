import sys
import itertools
import statistics
import numpy as np
import pandas as pd

def fishers_exact_test(df, statistic):
    """
    Performs Fishers exact test ...

    :param df: original dataframe
    :param statistic: statistic used to evaluate treatment effect
    :return: true probability of ...
    """
    # todo: could always just take M sample permutations, make as option
    # todo: what do I return?

    test_stat = statistic(df, 'outcome', 'W')

    t_indeces = [list(t_indeces) for t_indeces in itertools.combinations(range(len(df)), df['W'].sum())]
    t_indeces.remove(df.query('W == 1').index.tolist())
    stat_distribution = []
    for i in t_indeces:
        df['W'] = 0
        df.loc[i, 'W'] = 1
        stat_distribution.append(statistic(df, 'outcome', 'W'))

    return np.sum(np.where(stat_distribution > test_stat, 1, 0)) / len(stat_distribution)

if __name__ == '__main__':
    N = 10
    np.random.seed(2)
    columns = ['outcome']
    df = pd.DataFrame(np.random.uniform(-1, 1, size=(N, len(columns))), columns=columns)
    df['W'] = np.random.randint(0, 2, size=(N, 1))

    stat = statistics.abs_ave_by_treatment

    fishers_exact_test(df, stat)
