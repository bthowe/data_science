import sys
import numpy as np
import pandas as pd
import seaborn as sns  # todo: will need to add sklearn to (or maybe just this module) to setup.py
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression  # todo: will need to add sklearn to (or maybe just this module) to setup.py\

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 40000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)



def data_in():
    return pd.read_csv('/Users/thowe/Downloads/RawData.csv').\
        assign(
            time=lambda x: pd.to_datetime(x['TransDateTime']),
            hour=lambda x: x['time'].dt.hour,
            minute=lambda x: x['time'].dt.minute,
        ).\
        sort_values('time', ascending=True)


def _alpha_model_fit(df):
    model = LinearRegression()
    model.fit(df[['log_outcome']], df['log_p'])
    return model
    # return model.coef_[0]

def _alpha_estimate(df, outcome, threshold):
    df_temp = df.\
        assign(
            log_outcome=lambda x: np.log10(x[outcome]),
            log_p=lambda x: np.log10(x['p'])
        )

    model = _alpha_model_fit(
        df_temp.\
            query(f'log_outcome >= {threshold}').\
            query('log_outcome == log_outcome')
    )

    return model.coef_[0], model.intercept_


def alpha(df, outcome, threshold):
    return -_alpha_estimate(df, outcome, threshold)[0]


def log_log_plot(df, outcome, outcome_label, threshold=None):
    """Makes a log-log plot, also known as a Zipf plot. A variable representing the survival probability needs to be in
    the dataframe and named 'p'.

    threshold: float or None
        If not None, then will calculate and plot the slope of the tail.
    """
    # todo: error if no column names p

    sns.set_style("whitegrid")  # , {'axes.grid': False})
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(df[outcome], df['p'])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(outcome_label)
    ax.set_ylabel('$P_{>|X|}$')

    if threshold:
        a, b = _alpha_estimate(df, outcome, threshold)
        ax.plot(df[outcome], (df[outcome] ** a) * (10 ** b), color='firebrick')

    plt.title(f'Log-Log Plot of {outcome_label}: $\\alpha = {round(-a, 2)}$')
    plt.savefig('/Users/thowe/Downloads/zipf.png')
    plt.show()


def maximum_to_sum_plot(df_in, outcome, outcome_label, moment=4):
    """Maximum-to-Sum plot. Test 4 in chapter 10, page 192"""
    df = df_in.sample(50_000)
    df[outcome] = df[outcome] / 1000

    n = df.shape[0]
    ms_lst = []
    for num in range(1, n + 1):
        print(num)
        df_temp = df[outcome].iloc[0:num] ** moment
        ms_lst.append(df_temp.max() / df_temp.sum())
    print(df_temp)
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(n), ms_lst)
    ax.set_xlabel('n')
    ax.set_ylabel(f'MS({moment})')

    plt.title(f'Maximum-to-Sum Plot for {outcome_label}')
    plt.savefig('/Users/thowe/Downloads/ms_plot.png')
    plt.show()


def log_log_overall():
    df = data_in().\
        sort_values('RespTime', ascending=False). \
        reset_index(drop=True). \
        assign(p=lambda x: (x.index + 1) / x.shape[0])
    log_log_plot(df, 'RespTime', 'Response Time', threshold=4.4)
    print(alpha(df, 'RespTime', threshold=4.4))


def log_log_8_to_6():
    # going to abort this and just say, looks like there might be multiple processes
    df = data_in().\
        query('8 <= hour <= 18').\
        sort_values('RespTime', ascending=False). \
        reset_index(drop=True). \
        assign(p=lambda x: (x.index + 1) / x.shape[0])
    log_log_plot(df, 'RespTime', 'Response Time', threshold=4.4)
    print(alpha(df, 'RespTime', threshold=4.4))


def ms_plot():
  data_in().pipe(maximum_to_sum_plot, 'RespTime', 'Response Time')


def main():
    log_log_overall()
    # # log_log_8_to_6()

    ms_plot()


if __name__ == '__main__':
    main()
