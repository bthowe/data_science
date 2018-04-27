import sys
import patsy
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from statsmodels.regression.quantile_regression import QuantReg

# http://avesbiodiv.mncn.csic.es/estadistica/curso2011/qr6.pdf
# http://www.statsmodels.org/devel/examples/notebooks/generated/quantile_regression.html

def data_create():
    data = sm.datasets.engel.load_pandas().data
    return data

def least_absolute_deviation_model_create(df):
    mod = smf.quantreg('foodexp ~ income', df)
    return mod

def ols_model_create(df):
    mod = smf.ols('foodexp ~ income', df)
    return mod

def _parameter_lst_create(mod, quantile=None):
    if quantile:
        res = mod.fit(q=quantile)
    else:
        res = mod.fit()
    return [quantile, res.params['Intercept'], res.params['income']] + res.conf_int().ix['income'].tolist()

def parameter_data_create(q_mod, ols_mod):
    model_params = [_parameter_lst_create(q_mod, quantile) for quantile in np.arange(.05, .96, .1)]
    model_params.append(_parameter_lst_create(ols_mod))

    return pd.DataFrame(model_params, columns=['quantile', 'intercept', 'b1', 'lower_bound', 'upper_bound'])

def _line_plots(df, x, ax):
    y = lambda b0, b1: b0 + b1 * x

    for row in df.iloc[:-1, :].iterrows():
        ax.plot(x, y(row[1]['intercept'], row[1]['b1']), linestyle='dotted', color='grey')
    ax.plot(x, y(df.iloc[-1]['intercept'], df.iloc[-1]['b1']), color='red', label='OLS')
    ax.legend()

def _scatter_plot(data, ax):
    ax.scatter(data.income, data.foodexp, alpha=.2)

def data_plot(data, df_params):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    _line_plots(
        df_params,
        np.linspace(data['income'].min(), data['income'].max(), 100),
        ax
    )
    _scatter_plot(
        data,
        ax
    )
    plt.show()

def df_plot(df):
    df_quan = df.iloc[:-1]
    df_ols = df.iloc[-1:]

    n = len(df_quan)
    x = df_quan['quantile']

    fig=plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, df_quan['b1'], color='black', label='Quantile Reg.')
    ax.plot(x, df_quan['upper_bound'], color='black', label='upper bound', linestyle='dotted')
    ax.plot(x, df_quan['lower_bound'], color='black', label='lower bound', linestyle='dotted')
    ax.plot(x, [float(df_ols['b1'])] * n, color='red', label='OLS')
    ax.plot(x, [float(df_ols['upper_bound'])] * n, color='red', linestyle='dotted')
    ax.plot(x, [float(df_ols['lower_bound'])] * n, color='red', linestyle='dotted')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    data = data_create()
    quantile_model = least_absolute_deviation_model_create(data)
    ols_model = ols_model_create(data)

    df = parameter_data_create(quantile_model, ols_model)

    data_plot(data, df)
    df_plot(df)
