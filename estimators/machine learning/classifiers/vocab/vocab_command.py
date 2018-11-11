import sys
import joblib
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
from collections import defaultdict
from flask import Flask, request, render_template, jsonify, redirect

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


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

def observation_filter(df):
    return df.\
        query('timestamp > "{}"'.format(date)). \
        reset_index(drop=True)

def feature_create(df):
    df['active'] = 1
    df['date'] = df['timestamp'].dt.date  #.astype(str)
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    def day_features(x):
        x['Ltimestamp'] = x['timestamp'].shift(1)
        x['Lpage'] = x['page'].shift(1)
        x['Lbutton'] = x['button'].shift(1)

        x['time_since_last_button'] = (x['timestamp'] - x['Ltimestamp']).dt.seconds
        return x

    return df.groupby(df['date']).apply(day_features)  #.query('q1 < time_elapsed < q99')

def target_create(df):
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


def objects_create(df):
    X = df
    y = helpers.target_create(X).pop('target')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)  # , random_state=2)
    svg.upload_file_pickle(X_train, sub_directory=constants.data_directory,
                           file_name='X_train_{}.pkl'.format(timestamp), bucket=svg.bucket, name='pretention')
    svg.upload_file_pickle(y_train, sub_directory=constants.data_directory,
                           file_name='y_train_{}.pkl'.format(timestamp), bucket=svg.bucket, name='pretention')
    svg.upload_file_pickle(X_test, sub_directory=constants.data_directory, file_name='X_test_{}.pkl'.format(timestamp),
                           bucket=svg.bucket, name='pretention')
    svg.upload_file_pickle(y_test, sub_directory=constants.data_directory, file_name='y_test_{}.pkl'.format(timestamp),
                           bucket=svg.bucket, name='pretention')


def model_train(timestamp):
    X = joblib.load(constants.data_directory + '/X_train_{}.pkl'.format(timestamp))
    y = joblib.load(constants.data_directory + '/y_train_{}.pkl'.format(timestamp))

    pipeline = Pipeline(
        [
            ('feature_create', FunctionTransformer(helpers.feature_create, validate=False)),
            ('feature_drop', FunctionTransformer(helpers.feature_drop, validate=False)),
            ('feature_dummy_create', helpers.Dummies_Create()),
            ('fill_nulls', Imputer()),
            ('vt', VarianceThreshold()),
            ('select', RFE(XGBClassifier(n_estimators=10), step=50)),
            ('xgb', XGBClassifier(n_estimators=100))
        ]
    )

    # grid_search = GridSearchCV(pipeline, constants.param_grid, scoring='roc_auc', verbose=10, n_jobs=1, cv=3)
    grid_search = GridSearchCV(pipeline, constants.xgb_parameters, scoring='roc_auc', verbose=10, n_jobs=6, cv=3)


def scratch():
    X['time_elapsed_L1'] = X['time_elapsed'].shift(1) + X['iqr']

    # X.query('time_elapsed < total_time + 10', inplace=True)

    print(X['total_time'].groupby(X['date']).mean())

    print(X.query('date == "2018-10-26"'))
    # print(X.query('name == "Calvin"'))

# approaches
# 1. remove the last 1%
# 2. drop if later than time elapsed plus interquarter range of time elapsed
# 3. find the distribution of time between clicks and drop if significantly after the 99th percentile.
# 4. I could label all of the days and then create a model to predict whether the student was done at that point.
# 5. I could simply go throw and label all of the clicks that are no good.
# 6. Or just go before the last two which are main menu and then quit.
# 7. look for main menu and then quit and only take those prior.
    # -but what if there is a day in which there is not a quit? What do I do in that case? It seems I could simply
    # I could create observations in that have the previous click button and the subsequent click button as well as the time difference in time elapsed.
    # The labels could be a binary, whether I was done or not. The features would be 1. hour of day, 2. time since last click, 3. page, 4. button, 5.
    #
# 8. Go through the database and delete the garbage.
# 9. I probably favor that one, then. But, what do I do with ones that come later?
# 10.

# Still active or not
#

# what should this data set look like?
#

# target is if active or not
# going to have to drop the first observation out of every date


if __name__ == '__main__':
    date = '2018-09-01'
    # data_pull().\
    #     pipe(observation_filter).\
    #     pipe(feature_create)


    target_create(joblib.load('df.pkl'))
