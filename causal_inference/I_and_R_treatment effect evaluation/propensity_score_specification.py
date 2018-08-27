import sys
import joblib
import datetime
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import log_loss

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def _feature_keep(df):
    covars = ['treatment_group', 'monthly_premium', 'agent_tenure', 'policy_wait', 'state_id', 'payment_time']
    return df[covars]

def _feature_dummy_create(df):
    df['state_id'] = df['state_id'].astype(str)
    return pd.get_dummies(df)

def preprocess(df):
    return df.pipe(_feature_keep).pipe(_feature_dummy_create)

def additional_linear_terms(df, basic_covars, t_true):
    lr = LogisticRegression()
    lr.fit(df[basic_covars], t_true)
    t_pred = lr.predict_proba(df[basic_covars])
    l0 = log_loss(t_true, t_pred)

    other_covars = [covar for covar in df if covar not in basic_covars]
    score = {'score': 0, 'covar': ''}
    for covar in other_covars:
        new_covars = basic_covars + [covar]
        lr = LogisticRegression()
        lr.fit(df[new_covars], t_true)
        t_pred = lr.predict_proba(df[new_covars])
        l1 = log_loss(t_true, t_pred)

        new_score = l0 / l1
        if score['score'] < new_score:
            score['score'] = new_score
            score['covar'] = covar
    print(score)
    sys.exit()


def quadratic_interaction_terms(df):
    pass

if __name__ == '__main__':
    df = pd.read_csv('test_data.csv').pipe(preprocess)
    t = df.pop('treatment_group')

    basic_covariates = ['monthly_premium', 'policy_wait']
    df.\
        pipe(additional_linear_terms, basic_covariates, t).\
        pipe(quadratic_interaction_terms)
