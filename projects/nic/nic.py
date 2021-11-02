import sys
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm
import matplotlib.pyplot as plt

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
            month=lambda x: x['time'].dt.month,
            day=lambda x: x['time'].dt.day,
            hour=lambda x: x['time'].dt.hour,
            minute=lambda x: x['time'].dt.minute,
            second=lambda x: x['time'].dt.second,
            dow=lambda x: x['time'].dt.weekday,
            weekend=lambda x: x['dow'].apply(lambda y: 1 if y in [5, 6] else 0)
        ).\
        sort_values('time', ascending=True). \
        assign(time_between=lambda x: (x['time'] - x['time'].shift()).dt.seconds)


def hist_plotter(df):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(df['RespTime'], bins=100)
    plt.savefig('/Users/thowe/Downloads/hist.png')
    plt.show()


def confidence_intervals(df):
    print(np.mean(df['RespTime']))
    print(np.median(df['RespTime']))
    print(df['RespTime'].describe())
    print(st.norm.interval(0.975, loc=np.mean(df['RespTime']), scale=st.sem(df['RespTime'])))
    print(st.norm.interval(0.99, loc=np.mean(df['RespTime']), scale=st.sem(df['RespTime'])))


def median_ols(df):
    print(df.head())
    df_agg = df[['RespTime', 'minute', 'hour']].groupby(['minute', 'hour']).agg(['count', 'median']).\
        reset_index(drop=False).\
        assign(
            quarter_binary=lambda x: x['minute'].apply(lambda y: 1 if y in [0, 15, 30, 45] else 0),
            quarter_traffic_interaction=lambda x: x['RespTime']['count'] * x['quarter_binary'],
            outcome=lambda x: x['RespTime']['median'],
            traffic=lambda x: x['RespTime']['count'],
            intercept=1
        ) \
        [['outcome', 'intercept', 'traffic', 'quarter_binary', 'quarter_traffic_interaction']]
    print(df_agg.head(10))
    print(df_agg.tail(10))

    model = sm.OLS(df_agg['outcome'], df_agg[['intercept', 'traffic', 'quarter_binary', 'quarter_traffic_interaction']])
    results = model.fit()
    print(results.summary())


def max_ols(df):
    print(df.head())
    df_agg = df[['RespTime', 'minute', 'hour']].groupby(['minute', 'hour']).agg(['count', 'max']).\
        reset_index(drop=False).\
        assign(
            quarter_binary=lambda x: x['minute'].apply(lambda y: 1 if y in [0, 15, 30, 45] else 0),
            quarter_traffic_interaction=lambda x: x['RespTime']['count'] * x['quarter_binary'],
            outcome=lambda x: x['RespTime']['max'],
            traffic=lambda x: x['RespTime']['count'],
            intercept=1
        ) \
        [['outcome', 'intercept', 'traffic', 'quarter_binary', 'quarter_traffic_interaction']]
    print(df_agg.head(10))
    print(df_agg.tail(10))

    model = sm.OLS(df_agg['outcome'], df_agg[['intercept', 'traffic', 'quarter_binary', 'quarter_traffic_interaction']])
    results = model.fit()
    print(results.summary())


def diags(df):
    print(df.sort_values('RespTime', ascending=False).head(1000))

    # print(df[['RespTime', 'minute']].groupby('minute').agg(['count', 'median']))
    # print(df[['RespTime', 'hour']].groupby('hour').agg(['count', 'median']))
    # print(df[['RespTime', 'day']].groupby('day').agg(['count', 'median']))


    # pd.crosstab(index=df['hour'], columns=df['minute']).to_csv('/Users/thowe/Downloads/pivot_count.csv')
    # pd.pivot_table(df, values='RespTime', index='hour', columns='minute', aggfunc=np.max).to_csv('/Users/thowe/Downloads/pivot_max.csv')
    # pd.pivot_table(df, values='RespTime', index='hour', columns='minute', aggfunc=np.median).to_csv('/Users/thowe/Downloads/pivot.csv')

# todo: correlation between count by hour of day and response time, median and max?
#   traffic causes increase in response times. (might it be the opposite?)
# todo: there are regularities at 0, 15, 30, and 45.
# todo: other spikes that are unpredicatable given this data. What drives these? What are the consequences?

# todo: regress RespTime on and (0, 15, 30, 45)-binary
#   data: row representing hour and minute; columns (0, 15, 30, 45)-binary, count, interaction of the two, ...and outcome (1) median RespTime, and (2) max RespTime



#     need other descriptions of distributions.
# how do I figure out if there is some regularity in quarter hour verses not extreme values. Crosstab...regression...what should I do?

# what about other aspects of the distribution?
def agg_plotter(df, covar):
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(1, 1, 1)
    sns.barplot(x=covar, y='RespTime', data=df, ax=ax, estimator=np.max)
    # sns.barplot(x=covar, y='RespTime', data=df, ax=ax, estimator=np.median, ci=99)
    plt.show()

    # df_agg = df[['RespTime', covar]].\
    #     groupby(covar).median().\
    #     reset_index()
    #
    # sns.set_style("whitegrid")
    # fig = plt.figure(figsize=(20, 8))
    # ax = fig.add_subplot(1, 1, 1)
    # sns.barplot(x=covar, y='RespTime', data=df_agg, ax=ax, ci='sd')
    # plt.show()


def time_series_overall_plotter(df):
    df = df.sort_values('time', ascending=True)
    print(df.head(10))

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df['time'], df['RespTime'])
    plt.savefig('/Users/thowe/Downloads/time_series_overall.png')
    plt.show()


def time_series_day_plotter(df):
    # df = df.query('(month == 6) and (day == 11) and (hour == 3)').sort_values('time', ascending=True)
    df = df.query('(month == 6) and (day == 1)').sort_values('time', ascending=True)
    print(df.head(10))

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df['time'], df['RespTime'])
    # plt.savefig('/Users/thowe/Downloads/time_series_day.png')
    plt.show()

def time_series_hour_plotter(df):
    # average time between
    # df = df.sort_values('time', ascending=True)
    # df = df.query('"2017-06-01" <= time < "2017-06-30"').sort_values('time', ascending=True)
    # df = df.query('"2017-06-01" <= time < "2017-06-02" and (hour == 21)').sort_values('time', ascending=True)
    # df = df.query('"2017-07-24" <= time <= "2017-07-31"').sort_values('time', ascending=True)
    # df = df.query('(month == 6)').sort_values('time', ascending=True)
    # df = df.query('(month == 6) and (day == 11)').sort_values('time', ascending=True)
    # df = df.query('(month == 6) and (day == 11) and (hour == 7)').sort_values('time', ascending=True)
    df = df.query('(month == 6) and (day == 1) and (hour == 9)').sort_values('time', ascending=True)
    print(df.head(10))

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df['time'], df['RespTime'])
    plt.savefig('/Users/thowe/Downloads/time_series_hour.png')
    plt.show()

def time_between(df_in):
    df = df_in.query('(month == 6) and (day == 6) and (hour == 9)')

    print(df[['minute', 'time_between']].groupby('minute').mean())

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df['minute'], df['time_between'])
    plt.show()

def count_per_minute(df_in):
    df = df_in.query('(month == 6) and (day == 6) and (hour == 9)')

    print(df[['RespTime', 'minute']].groupby('minute').count())

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df['minute'], df['RespTime'])
    plt.show()

def count_by_date_plotter(df):
    # todo: number doesn't change over time, but is seasonality weekly
    df = df.\
        assign(date=lambda x: x['time'].dt.date).\
        sort_values('time', ascending=True) \
        [['time', 'date']].\
        groupby('date').count().\
        reset_index()
    print(df.head(10))
    # sys.exit()

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df['date'], df['time'])
    plt.show()

def time_between_plotter(df_in):
    # todo: the only abnormality looks like 6/11
    df = df_in.sort_values('time', ascending=True).\
        query('(month == 6) and (day == 6) and (hour == 9)')
    print(df.head(10))
    # sys.exit()

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df['time'], df['time_between'])
    plt.show()

    print(df.query('"2017-06-11" <= time < "2017-06-12"').query('time_between > 100'))
#     I don't exactly know what to make of this.


# todo: are some days worse? does it not happen some days?

def main():
    df = data_in()

    # hist_plotter(df)
    confidence_intervals(df)
    sys.exit()

    # time_series_overall_plotter(df)
    # time_series_day_plotter(df)
    # time_series_hour_plotter(df)

    # count_by_date_plotter(df)  # use this?
    # time_between_plotter(df)  # use this?
    # time_between(df)  # use this?

    # median_ols(df)
    # max_ols(df)

    diags(df)

    # for covar in ['minute', 'day']:
    for covar in ['month', 'hour', 'minute', 'weekend', 'dow', 'day']:
        agg_plotter(df, covar)



if __name__ == '__main__':
    main()

# print(df.info())
# print(df.head())
# print(df.tail())
# print(df.describe())

# mean = np.mean(df['RespTime'])
# median = np.median(df['RespTime'])
# mad = np.mean(np.abs(median - df['RespTime']))
# print(mean)
# print(median)
# print(mad)
#
#
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(1, 1, 1)
# ax.hist(df['RespTime'], bins=25)
# plt.show()




