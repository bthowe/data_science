import sys
import joblib
import yagmail
import datetime
import numpy as np
import pandas as pd
import SVGAnalytics as svg
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
import xgbclassifier_constants

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def preprocess_raw_data(timestamp):
    ed = datetime.datetime.today() - datetime.timedelta(days=14)
    sd = datetime.datetime.today() - datetime.timedelta(days=744)
    # app_id_list = helpers.data_pull(sd, ed, 1).drop_duplicates(subset=['pretension_app_id'])['pretension_app_id'].values.tolist()
    # raw_start = helpers.feature_pull(tuple(map(str, app_id_list)))
    raw_start = helpers.feature_pull(sd=sd, ed=ed)
    svg.upload_file_csv(
        raw_start,
        sub_directory=constants.data_directory,
        file_name='raw_start_{}.csv'.format(timestamp),
        bucket=svg.bucket,
        name='pretention'
    )

def preprocess_objects_create(timestamp):
    X = pd.read_csv(constants.data_directory + '/raw_start_{}.csv'.format(timestamp)).\
        pipe(helpers.feature_type).\
        pipe(helpers.observation_filter)
    y = helpers.target_create(X).pop('target')

    return train_test_split(X, y, test_size=0.33)  #, random_state=2)

def model_train(X_train, X_test, y_train, y_test):
    X = joblib.load(constants.data_directory + '/X_train_{}.pkl'.format(timestamp))
    y = joblib.load(constants.data_directory + '/y_train_{}.pkl'.format(timestamp))

    pipeline = Pipeline(
        [
            ('feature_create', FunctionTransformer(helpers.feature_create, validate=False)),
            ('feature_drop', FunctionTransformer(helpers.feature_drop, validate=False)),
            ('feature_dummy_create', helpers.Dummies_Create()),
            ('fill_nulls', Imputer()),
            ('vt', VarianceThreshold()),
            ('select', RFE(RandomForestClassifier(), step=20)),
            ('classifier', XGBClassifier(random_state=42))
        ]
    )
    grid_search = GridSearchCV(pipeline, xgbclassifier_constants.param_grid, scoring='roc_auc', verbose=10, n_jobs=6, cv=3)
    # grid_search = RandomizedSearchCV(pipeline, xgbclassifier_constants.param_grid_randomized, n_iter=1000, scoring='roc_auc', verbose=10, n_jobs=-1, cv=3)  # n_iter is the number of unique parameter values, thus, 3000 fits will be made here.

    grid_search.fit(X, y)
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    # svg.upload_file_pickle(grid_search, sub_directory=constants.data_directory, file_name='best_model.pkl', bucket=svg.bucket, name='pretention')
    svg.upload_file_pickle(grid_search, sub_directory=constants.data_directory, file_name='best_model_{}.pkl'.format(timestamp), bucket=svg.bucket, name='pretention')



# randomsearch