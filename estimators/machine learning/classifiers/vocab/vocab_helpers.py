import sys
import datetime
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
from pymongo import MongoClient
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import uniform

def data_pull():
    client = MongoClient()
    db_vocab = client['vocab']

    cols = ['_id', 'button', 'name', 'page', 'timestamp']

    df_main = pd.DataFrame(list(db_vocab['Main'].find({'name': 'Calvin'})))
    df_main['timestamp'] = df_main['timestamp'].apply(datetime.datetime.fromtimestamp)

    df_practice = pd.DataFrame(list(db_vocab['Practice'].find({'name': 'Calvin'})))
    df_practice['timestamp'] = df_practice['timestamp'].apply(datetime.datetime.fromtimestamp)

    df_quiz = pd.DataFrame(list(db_vocab['Quiz'].find({'name': 'Calvin'})))
    df_quiz['timestamp'] = df_quiz['timestamp'].apply(datetime.datetime.fromtimestamp)

    return df_main[cols]. \
        append(df_practice[cols]). \
        append(df_quiz[cols]). \
        sort_values('timestamp')   #. \



def observation_filter_train(df, start_date, end_date):
    return df.\
        query('timestamp > "{}"'.format(start_date)). \
        query('timestamp < "{}"'.format(end_date)). \
        reset_index(drop=True)

def observation_filter(df, end_date):
    return df.\
        query('timestamp >= "{}"'.format(end_date)). \
        reset_index(drop=True)

# def objects_create(df):
#     return target_create(df)
#     todo: do I need a test and train set?

def target_create_train(df):
    df['active'] = 1

    df.at[list(range(217, 224)), 'active'] = 0
    df.at[505, 'active'] = 0
    df.at[506, 'active'] = 0
    df.at[1427, 'active'] = 0
    df.at[1921, 'active'] = 0
    df.at[1922, 'active'] = 0
    df.at[3504, 'active'] = 0
    df.at[3505, 'active'] = 0
    df.at[4069, 'active'] = 0
    df.at[4070, 'active'] = 0
    df.at[6682, 'active'] = 0
    df.at[6683, 'active'] = 0
    df.at[10106, 'active'] = 0
    df.at[list(range(11296, 11302)), 'active'] = 0
    df.at[11862, 'active'] = 0
    return df.reset_index(drop=True)

def target_create(df):
    df['active'] = 1
    return df.reset_index(drop=True)


def feature_create(df):
    df.loc[:, 'date'] = df['timestamp'].dt.date  #.astype(str)
    df.loc[:, 'hour_of_day'] = df['timestamp'].dt.hour.astype(str)
    df.loc[:, 'day_of_week'] = df['timestamp'].dt.dayofweek.astype(str)

    df.loc[:, 'Ltimestamp'] = df['timestamp'].shift(1)
    df.loc[:, 'Lpage'] = df['page'].shift(1)
    df.loc[:, 'Lbutton'] = df['button'].shift(1)
    df.loc[:, 'Lhour_of_day'] = df['hour_of_day'].shift(1)
    df.loc[:, 'Lday_of_week'] = df['day_of_week'].shift(1)

    df.loc[:, 'time_since_last_button'] = (df['timestamp'] - df['Ltimestamp']).dt.seconds

    df = df.assign(new_day=0)
    def day_features(x):
        x['new_day'].iloc[0] = 1
        return x

    df = df.groupby(df['date']).apply(day_features).iloc[1:]
    return df


def feature_none(df):
    df.at[df['page'] != df['page'], 'page'] = 'None'
    df.at[df['button'] != df['button'], 'button'] = 'None'
    df.at[df['hour_of_day'] != df['hour_of_day'], 'hour_of_day'] = 'None'
    df.at[df['day_of_week'] != df['day_of_week'], 'day_of_week'] = 'None'
    return df


def feature_drop(df):
    keep = ['button', 'page', 'hour_of_day', 'day_of_week', 'Lpage', 'Lbutton', 'time_since_last_button', 'Lhour_of_day', 'Lday_of_week', 'new_day']
    return df[keep]

class Dummies_Create(BaseEstimator, TransformerMixin):
    """This class creates dummy variables in a dataset. The fit method creates the dictvectorizer object, and the
    transform method uses it to transform the dataset according to the initial data. """

    def __init__(self):
        self.v = DictVectorizer(sparse=False)

    def transform(self, df, **transform_params):
        return pd.DataFrame(self.v.transform(row for _, row in df.iterrows()), columns=self.v.feature_names_)

        # df_to_dict = df.to_dict('records')
        # return pd.DataFrame(self.v.transform(df_to_dict), columns=self.v.feature_names_)

    def fit(self, df, y=None, **fit_params):
        self.v.fit(row for _, row in df.iterrows())
        return self


# todo: Lbutton etc. needs to be dropped

