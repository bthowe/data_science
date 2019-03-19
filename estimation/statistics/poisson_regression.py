import sys
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import binom, uniform
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.genmod.families import links as L
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def data_create_train():
    X_train = joblib.load('data_files/poisson_data.pkl').reset_index(drop=True)
    y_train = joblib.load('data_files/poisson_outcomes.pkl').reset_index(drop=True)
    df_train = pd.concat(
        [
            X_train,
            y_train
        ],
        axis=1
    )
    X_test = joblib.load('data_files/poisson_data_test.pkl').reset_index(drop=True)
    y_test = joblib.load('data_files/poisson_outcomes_test.pkl').reset_index(drop=True)
    df_test = pd.concat(
        [
            X_test,
            y_test
        ],
        axis=1
    )
    return df_train, df_test


class Glm(BaseEstimator, ClassifierMixin):
    def __init__(self, link=sm.families.Gaussian(), alpha=None, l1_weight=None):
        self.alpha = alpha
        self.l1_weight = l1_weight
        self.link = link

        self.model = None
        self.ss = StandardScaler()

    def fit(self, X, y=None):
        X_ss = self.ss.fit_transform(X)

        gamma_model = sm.GLM(y, X_ss, family=self.link)
        if self.alpha:
            self.model = gamma_model.fit_regularized(alpha=self.alpha, L1_wt=self.l1_weight)
        else:
            self.model = gamma_model.fit()
        return self

    def predict(self, X, y=None):
        X_ss = self.ss.transform(X)
        pred = self.model.predict(X_ss)
        return pred

def glm_reg_cv(df, covars, target):
    params = {
        'alpha': uniform(0, 10),
        'l1_weight': uniform(),
        'link': [sm.families.Poisson(), sm.families.NegativeBinomial()]
    }

    grid_search = RandomizedSearchCV(Glm(), params, n_iter=150, scoring='neg_mean_squared_error', verbose=10, n_jobs=1, cv=3)
    grid_search.fit(df[covars], df[target])
    print('Poisson Regression best parameters: {}'.format(grid_search.best_params_))
    print('Poisson Regression best score: {}'.format(grid_search.best_score_))
    return grid_search

def normal_reg(df, covars, target):
    params = {
        'alpha': uniform(),
        'fit_intercept': [True, False],
        'normalize': [True, False]
    }

    grid_search = RandomizedSearchCV(Ridge(), params, n_iter=150, scoring='neg_mean_squared_error', verbose=10, n_jobs=1, cv=3)
    grid_search.fit(df[covars], df[target])
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    return grid_search

def model_validation(df, model_dict, covars, target):
    for model_name, model in model_dict.items():
        y_pred = model.predict(df[covars])
        y_true = df[target]
        print('{0} MSE: {1}'.format(model_name, np.mean((y_pred - y_true) ** 2)))

def plotter(df, model_dict, covars):
    count = 0
    fig = plt.figure(figsize=(12, 8))
    for model_name, model in model_dict.items():
        count += 1
        df['y_pred'] = model.predict(df[covars])
        for bucket in [('Boberdoo (fixed)', 'sandybrown'), ('Boberdoo (dynamic)', 'cornflowerblue'), ('SciOps', 'seagreen'), ('first_party', 'firebrick')]:
            ax = fig.add_subplot(len(model_dict), 1, count)
            df_temp = df.loc[df['bucket={}'.format(bucket[0])] == 1]
            df_temp.sort_values('spend', inplace=True)
            ax.scatter(df_temp['spend'], df_temp['ltv'], alpha=0.25, color=bucket[1], label='{} actual'.format(bucket[0]))
            ax.plot(df_temp['spend'], model.predict(df_temp[covars]), color=bucket[1], label='{} predicted'.format(bucket[0]))

        plt.xlabel('Spend')
        plt.ylabel('Apps')
        plt.title('Apps vs Spend, test set, {}'.format(model_name))
    plt.savefig('/Users/travis.howe/Downloads/apps_vs_Spend_by_bucket.png')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    covars = ['bucket=Boberdoo (dynamic)', 'bucket=Boberdoo (fixed)', 'bucket=SciOps', 'bucket=first_party', 'spend', 'spend2', 'spend_bucket=Boberdoo (dynamic)', 'spend2_bucket=Boberdoo (dynamic)', 'spend_bucket=Boberdoo (fixed)', 'spend2_bucket=Boberdoo (fixed)', 'spend_bucket=SciOps', 'spend2_bucket=SciOps', 'spend_bucket=first_party', 'spend2_bucket=first_party']
    df_train, df_test = data_create_train()

    gamma_model = sm.GLM(df_train['ltv'], df_train[covars], family=sm.families.NegativeBinomial())
    # self.model = gamma_model.fit_regularized(alpha=self.alpha, L1_wt=self.l1_weight)
    res = gamma_model.fit_regularized(alpha=0)
    res2 = gamma_model.fit()
    # todo: why is there a difference between these two?

    gamma_model = sm.GLM(df_train['ltv'], df_train[covars], family=sm.families.Poisson())
    # self.model = gamma_model.fit_regularized(alpha=self.alpha, L1_wt=self.l1_weight)
    pres = gamma_model.fit_regularized(alpha=0)
    pres2 = gamma_model.fit()

    # glm = Glm(sm.families.NegativeBinomial())
    # glm.fit(df_train[covars], df_train['ltv'])
    model_dict = {
        # 'Poisson': poisson_reg(df_train, covars, 'ltv'),
        # 'Negative Binomial': neg_bin_reg(df_train, covars, 'ltv'),
        # 'GLM_search': glm_reg_cv(df_train, covars, 'ltv'),
        # 'p_reg': pres,
        'p': pres2,
        # 'nb_reg': res,
        'nb': res2,
        'Normal': normal_reg(df_train, covars, 'ltv')
    }
    # sys.exit()
    #
    # joblib.dump(model_dict, 'model_dict.pkl')
    # model_dict = joblib.load('model_dict.pkl')
    model_validation(df_test, model_dict, covars, 'ltv')

    plotter(df_test, model_dict, covars)
