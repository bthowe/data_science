import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import os
os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def data_create():
    df = pd.read_csv('data_files/daily-minimum-temperatures-in-me.csv')
    df.columns = ['date', 'temp']
    df.set_index('date', inplace=True)
    return df

def corr_plots(df):
    df.plot()
    plot_acf(df, lags=50)
    plot_pacf(df, lags=50)
    plt.show()

def acf_calculate(df, lag):
    df['Ltemp'] = df['temp'].shift(lag)
    print(df.corr())

def pacf_calculate(df, lag):
    df['constant'] = 1
    for i in range(1, lag + 1):
        df['L{}temp'.format(i)] = df['temp'].shift(i)
    df.dropna(inplace=True)

    X = df
    y = df.pop('temp')

    mod = sm.OLS(y, X)
    res = mod.fit()
    print('The partial autocorrelation for lag {0} is {1}'.format(lag, round(res.params[-1], 3)))

def arima_forecast(df):
    r = robjects.r
    pandas2ri.activate()
    forecast = importr('forecast')

    ry_train = r.ts(df, start=r.c(1981, 1), frequency=365)

    arima_fit = forecast.auto_arima(ry_train)
    print(r.summary(arima_fit))
    forecast = forecast.forecast(arima_fit, h=180)

    df_forecast = pd.concat(
        [
            pd.DataFrame(np.array(forecast.rx('mean')).flatten(), columns=['prediction']),
            pd.DataFrame(np.array(forecast.rx('lower')[0])[:, 1], columns=['ci_lower_bound']),
            pd.DataFrame(np.array(forecast.rx('upper')[0])[:, 1], columns=['ci_upper_bound'])
        ],
        axis=1
    )
    print(df_forecast)
    return df_forecast

def arima_statsmodels(df):
    model = ARIMA(df, order=(1, 0, 1))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    forecast = model_fit.forecast(steps=180, alpha=0.05)

    df_forecast = pd.concat(
        [
            pd.DataFrame(forecast[0], columns=['prediction']),
            pd.DataFrame(forecast[2][:, 0], columns=['ci_lower_bound']),
            pd.DataFrame(forecast[2][:, 1], columns=['ci_upper_bound'])
        ],
        axis=1
    )
    print(df_forecast)
    return df_forecast

def pred_plotter(df_forecast, df_statsmodels):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(len(df_forecast)), df_forecast['prediction'], color='red', alpha=.5, label='Forecast Library Prediction')
    ax.fill_between(range(len(df_forecast)), df_forecast['ci_lower_bound'], df_forecast['ci_upper_bound'], alpha=0.2, color='red', label='Forecast 95% Confidence Interval')
    ax.plot(range(len(df_statsmodels)), df_statsmodels['prediction'], color='blue', alpha=.5, label='Stats Models Library Prediction')
    ax.fill_between(range(len(df_statsmodels)), df_statsmodels['ci_lower_bound'], df_statsmodels['ci_upper_bound'], alpha=0.2, color='blue', label='Stats Models 95% Confidence Interval')
    plt.legend()
    plt.show()

def plot_actual_and_forecast(df_actual, df_forecast):
    print(df_forecast); sys.exit()
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(len(df_actual)), df_actual['temp'], color='cornflowerblue', label='Actual')
    ax.plot(range(len(df_actual) + 1, len(df_actual) + 181), df_forecast['prediction'], color='red', label='Forecast')
    ax.fill_between(range(len(df_actual) + 1, len(df_actual) + 181), df_forecast['ci_lower_bound'], df_forecast['ci_upper_bound'], color='royalblue', alpha=0.2, label='95% Confidence Interval')
    ax.set_ylabel('Temp $C$')
    ax.set_xlabel('Time Periods')
    plt.show()

if __name__ == '__main__':
    df = data_create()

    acf_calculate(df, 3)
    pacf_calculate(df, 2)
    corr_plots(df)

    # df_f = data_create().pipe(arima_forecast)
    df_sm = data_create().pipe(arima_statsmodels)
    # pred_plotter(df_f, df_sm)
    plot_actual_and_forecast(data_create(), df_sm)
