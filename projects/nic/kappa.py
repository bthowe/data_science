import sys
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 40000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def data_in():
    return pd.read_csv('/Users/thowe/Downloads/RawData.csv').\
        assign(time=lambda x: pd.to_datetime(x['TransDateTime'])).\
        sort_values('time', ascending=True)

def m(data, n, bootstrap_n):
    df_n = np.empty(0)
    for _ in range(bootstrap_n):
        df_n = np.append(df_n, data.sample(n=n, replace=True).sum())
    return np.mean(np.abs(df_n - np.mean(df_n)))


# def kappa(data, n, n0=1, bootstrap_n=1_000_000, rescale=False):
    # if rescale:
    #     data = data * (np.sqrt(2 / np.pi) / m(data, 1, bootstrap_n))
def kappa(data, n, n0=1, bootstrap_n=1_000_000, rescale=False):
    m_n = m(data, n, bootstrap_n)
    m_n0 = m(data, n0, bootstrap_n)

    return 2 - (np.log(n) - np.log(n0)) / np.log(m_n / m_n0)


def n_v(k_1, n_g):
    return n_g ** (-1 / (k_1 - 1))


def k1_and_n_v_estimate(df):
    # find kappa_1
    k = kappa(df, n=2, n0=1)
    print(k)  # ~ 0.7865784097209034, 0.8067374433723662

    # find n_v
    n = n_v(k, 30)
    print(n)  # ~ 8_339_549, 43_962_297


def kappa_convergence_plot(df, n_lst):
    kappa_lst = [kappa(df, n=n, n0=1) for n in n_lst]

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(n_lst, kappa_lst)
    ax.set_ylim([0, .9])
    plt.savefig('/Users/thowe/Downloads/kappa_convergence_plot.png')
    plt.show()


def main():
    df = data_in()
    print(df.head())
    print(df.info())
    print(df.shape)  # 2_110_978

    k1_and_n_v_estimate(df['RespTime'])
    # kappa_convergence_plot(df['RespTime'], range(2, 1003, 50))

if __name__ == '__main__':
    main()
