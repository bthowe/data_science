import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from itertools import combinations_with_replacement
from sklearn.linear_model import LogisticRegression
import propensity_score_specification as prop

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def feature_create_log_odds_ratio(df):
    """
    Calculate the log odds ratio.
    """
    df['l'] = np.log(df['e'] / (1 - df['e']))
    return df


def trim(df):
    """
    Drop all control units with an estimated propensity score less than the smallest value of the estimated propensity score among the treated units.
    Drop all treated units with an estimated propensity score greater than the largest value of the estimated propensity score among the control units.
    """
    control_max = df.loc[df['t'] == 0]['e'].max()
    treatment_min = df.loc[df['t'] == 1]['e'].min()
    return df.\
        loc[~((df['t'] == 0) & (df['e'] < treatment_min))].\
        loc[~((df['t'] == 1) & (df['e'] > control_max))]


def adequate_balance(df, block_num):
    df.query('block == {}'.format(block_num), inplace=True)

    N_c = df.query('t == 0').shape[0]
    N_t = df.query('t == 1').shape[0]
    N = df.shape[0]

    l_c = df.query('t == 0')['l'].mean()
    l_t = df.query('t == 1')['l'].mean()

    S2 = (1 / (N - 2)) * (((df.query('t == 0')['l'] - l_c) ** 2).sum() + ((df.query('t == 1')['l'] - l_t) ** 2).sum())

    t = (l_t - l_c) / np.sqrt(S2 * ((1 / N_c) + (1 / N_t)))  # it is reasonable that l_t is greater than l_c
    return t > 1


def amenable_to_split(df, k, block_num):
    df.query('block == {}'.format(block_num), inplace=True)

    b_prime = df['e'].median()

    N_c_l = df.query('t == 0').query('e < {}'.format(b_prime)).shape[0]
    N_t_l = df.query('t == 1').query('e < {}'.format(b_prime)).shape[0]
    N_c_u = df.query('t == 0').query('{} <= e'.format(b_prime)).shape[0]
    N_t_u = df.query('t == 1').query('{} <= e'.format(b_prime)).shape[0]

    return (min(N_c_l, N_t_l, N_c_u, N_t_u) >= 3) and (min(N_c_l, N_t_l, N_c_u, N_t_u) >= k + 2)


def block_split(df, block_num, block_dic):
    df = df.query('block == {}'.format(block_num))
    b_prime = df['e'].median()

    block_num_new = max(list(block_dic.keys())) + 1
    block_dic[block_num_new] = [df['e'].median(), block_dic[block_num][1]]
    block_dic[block_num] = [block_dic[block_num][0], df['e'].median()]

    df.loc[df['e'] <= b_prime, 'block'] = block_num
    df.loc[df['e'] > b_prime, 'block'] = block_num_new

    return df, block_dic


def boundary_point_selection(df, k):
    df = trim(df)

    df['block'] = 0
    block_dic = {0: [df['e'].min(), df['e'].max()]}

    candidate_to_split = list(block_dic.keys())
    while candidate_to_split:
        print("candidate:", candidate_to_split)
        for block in candidate_to_split:
            print('block:', block)
            if adequate_balance(df, block) and amenable_to_split(df, k, block):
                print('yes')
                df, block_dic = block_split(df, block, block_dic)
                candidate_to_split.append(max(list(block_dic.keys())))
            else:
                print('no')
                candidate_to_split.remove(block)


if __name__ == '__main__':
    df = joblib.load('df.pkl').pipe(feature_create_log_odds_ratio)
    k = df.shape[1] - 2
    boundary_point_selection(df[['e', 't', 'l']], k)


# todo: test this with some real data
