import sys
import joblib
import datetime
import numpy as np
import pandas as pd
import vocab_helpers as helpers
import matplotlib.pyplot as plt
from pymongo import MongoClient
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

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
        query('timestamp > "{}"'.format(start_date)). \
        query('timestamp < "{}"'.format(end_date)). \
        reset_index(drop=True)

def objects_create(df):
    return helpers.target_create(df)
    # todo: do I need a test and train set?

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)  # , random_state=2)
    # svg.upload_file_pickle(X_train, sub_directory=constants.data_directory, file_name='X_train_{}.pkl'.format(timestamp), bucket=svg.bucket, name='pretention')
    # svg.upload_file_pickle(y_train, sub_directory=constants.data_directory, file_name='y_train_{}.pkl'.format(timestamp), bucket=svg.bucket, name='pretention')
    # svg.upload_file_pickle(X_test, sub_directory=constants.data_directory, file_name='X_test_{}.pkl'.format(timestamp), bucket=svg.bucket, name='pretention')
    # svg.upload_file_pickle(y_test, sub_directory=constants.data_directory, file_name='y_test_{}.pkl'.format(timestamp), bucket=svg.bucket, name='pretention')
    # return (X, y)

def model_train(X):
    y = X.pop('active')

    pipeline = Pipeline(
        [
            ('feature_create', FunctionTransformer(helpers.feature_create, validate=False)),
            ('feature_none', FunctionTransformer(helpers.feature_none, validate=False)),
            ('feature_drop', FunctionTransformer(helpers.feature_drop, validate=False)),
            ('feature_dummies', helpers.Dummies_Create()),
            # ('xgb', XGBClassifier(n_estimators=10))
        ]
    )

    print(pipeline.fit_transform(X, y).shape)
    print(y.describe())
    sys.exit()


    xgb_rand_params = {
        'xgb__max_depth': [3, 4, 5, 6, 7],
        'xgb__learning_rate': uniform(),
        'xgb__reg_lambda': uniform(),
        'xgb__min_child_weight': [10, 12, 14, 16, 18]
    }

    grid_search = RandomizedSearchCV(pipeline, xgb_rand_params, scoring='roc_auc', verbose=10, n_jobs=1, cv=3)
    grid_search.fit(X, y)
    print(grid_search.best_params_)
    print(grid_search.best_score_)



if __name__ == '__main__':
    start_date = '2018-09-01'
    end_date = '2018-11-09'
    data_pull().\
        pipe(observation_filter).\
        pipe(objects_create).\
        pipe(model_train)
