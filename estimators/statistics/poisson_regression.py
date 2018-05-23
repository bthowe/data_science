import sys
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import binom, uniform
from sklearn.linear_model import Ridge
from statsmodels.genmod.families import links as L
from sklearn.model_selection import RandomizedSearchCV

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def data_create_train():
    X_train = joblib.load('data_files/poisson_data.pkl')
    y_train = joblib.load('data_files/poisson_outcomes.pkl').reset_index(drop=True)
    df_train = pd.concat(
        [
            X_train,
            y_train
        ],
        axis=1
    )
    X_test = joblib.load('data_files/poisson_data_test.pkl')
    y_test = joblib.load('data_files/poisson_outcomes_test.pkl').reset_index(drop=True)
    df_test = pd.concat(
        [
            X_test,
            y_test
        ],
        axis=1
    )
    return df_train, df_test

def poisson_reg(df, covars, target):
    gamma_model = sm.GLM(df[target], df[covars], family=sm.families.Poisson())
    gamma_results = gamma_model.fit()
    print(gamma_results.summary())
    return gamma_results

def neg_bin_reg(df, covars, target):
    gamma_model = sm.GLM(df[target], df[covars], family=sm.families.NegativeBinomial())
    gamma_results = gamma_model.fit()
    print(gamma_results.summary())
    return gamma_results

def normal_reg(df, covars, target):
    params = {
        'alpha': uniform(),
        'fit_intercept': [True, False],
        'normalize': [True, False]
    }

    grid_search = RandomizedSearchCV(Ridge(), params, n_iter=150, scoring='neg_mean_absolute_error',
                                     verbose=10, n_jobs=1, cv=3)
    grid_search.fit(df[covars], df[target])
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    return grid_search

def plotter(df, model, model_name, covars):
    df['y_pred'] = model.predict(df[covars])

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)

    for bucket in [('Boberdoo (fixed)', 'sandybrown'), ('Boberdoo (dynamic)', 'cornflowerblue'), ('SciOps', 'seagreen'), ('first_party', 'firebrick')]:
        df_temp = df.loc[df['bucket={}'.format(bucket[0])] == 1]
        df_temp.sort_values('spend', inplace=True)
        ax.scatter(df_temp['spend'], df_temp['ltv'], alpha=0.25, color=bucket[1], label='{} actual'.format(bucket[0]))
        ax.plot(df_temp['spend'], model.predict(df_temp[covars]), color=bucket[1], label='{} predicted'.format(bucket[0]))
    # ax.plot(np.linspace(0, max(df['spend'].max(), df['ltv'].max()), 1000), np.linspace(0, max(df['spend'].max(), df['ltv'].max()), 1000), color='black', linestyle='--')

    plt.xlabel('Spend')
    plt.ylabel('Apps')
    plt.legend()
    plt.title('Apps vs Spend, test set, {}'.format(model_name))
    plt.savefig('/Users/travis.howe/Downloads/apps_vs_Spend_by_bucket_{}.png'.format(model_name))
    # plt.show()


if __name__ == '__main__':
    covars = ['bucket=Boberdoo (dynamic)', 'bucket=Boberdoo (fixed)', 'bucket=SciOps', 'bucket=first_party', 'spend', 'spend2', 'spend_bucket=Boberdoo (dynamic)', 'spend2_bucket=Boberdoo (dynamic)', 'spend_bucket=Boberdoo (fixed)', 'spend2_bucket=Boberdoo (fixed)', 'spend_bucket=SciOps', 'spend2_bucket=SciOps', 'spend_bucket=first_party', 'spend2_bucket=first_party']
    df_train, df_test = data_create_train()

    poisson = poisson_reg(df_train, covars, 'ltv')
    neg_bin = neg_bin_reg(df_train, covars, 'ltv')
    normal = normal_reg(df_train, covars, 'ltv')

    plotter(df_test, poisson, 'Poisson', covars)
    plotter(df_test, neg_bin, 'Negative Binomial', covars)
    plotter(df_test, normal, 'Normal', covars)
    plt.show()
