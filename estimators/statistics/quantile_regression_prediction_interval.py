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

def _fit_model(mod, quantile=None):
    if quantile:
        res = mod.fit(q=quantile)
    else:
        res = mod.fit()
    return res

# todo: can't I simply combine the fit step with the creation step above?
# todo: maybe work on combining this with the other script
# todo: how does this compare to a traditional prediction interval?
def predicted_data_create(X, q_mod, ols_mod):
    df = pd.DataFrame(_fit_model(ols_mod).predict(X['income']), columns=['mean'])
    for quantile in [.025, .975, .5]:
        df['quantile_{}'.format(quantile)] = _fit_model(q_mod, quantile).predict(X['income'])
    df['income'] = X['income']
    return df

def plotter(df, ordered=False):
    if ordered:
        df.sort_values('income', inplace=True)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df['income'], df['mean'], color='seagreen', label='Mean Prediction')
    ax.plot(df['income'], df['quantile_0.5'], color='cornflowerblue', label='Median Prediction')
    # ax.plot(df['income'], df['quantile_0.025'], color='olive', label='2.5 Quantile Prediction')
    # ax.plot(df['income'], df['quantile_0.975'], color='darkgoldenrod', label='97.5 Quantile Prediction')
    ax.fill_between(df['income'], df['quantile_0.025'], df['quantile_0.975'], alpha=0.2, color='r')
    ax.set_ylabel('Food Expenditures')
    ax.set_xlabel('Income')
    plt.title('95% Prediction Interval for Food Expenditures Conditional on Income')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    data = data_create()
    quantile_model = least_absolute_deviation_model_create(data)
    ols_model = ols_model_create(data)

    df = predicted_data_create(data, quantile_model, ols_model)

    # print(df.describe())
    # sys.exit()
    plotter(df, ordered=True)
