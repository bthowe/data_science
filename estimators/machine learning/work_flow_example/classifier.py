import sys
import joblib
import datetime
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFE
from sklearn.preprocessing import FunctionTransformer, Imputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from work_flow_example import classifier_constants, classifier_helpers

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def preprocess_raw_data():
    ed = datetime.datetime.today() - datetime.timedelta(days=14)
    sd = datetime.datetime.today() - datetime.timedelta(days=744)
    joblib.dump(classifier_helpers.feature_pull(sd=sd, ed=ed), 'raw_start.pkl')

def preprocess_objects_create():
    X = joblib.load('raw_start.pkl'). \
        pipe(classifier_helpers.feature_type). \
        pipe(classifier_helpers.observation_filter)
    y = classifier_helpers.target_create(X).pop('target')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=84)
    joblib.dump(X_train, 'X_train.pkl')
    joblib.dump(y_train, 'y_train.pkl')
    joblib.dump(X_test, 'X_test.pkl')
    joblib.dump(y_test, 'y_test.pkl')

def model_train():
    X = joblib.load('../data_files/X_train.pkl')
    y = joblib.load('../data_files/y_train_clsf.pkl')

    pipeline = Pipeline(
        [
            ('feature_create', FunctionTransformer(classifier_helpers.feature_create, validate=False)),
            ('feature_drop', FunctionTransformer(classifier_helpers.feature_drop, validate=False)),
            ('feature_dummy_create', classifier_helpers.Dummies_Create()),
            ('fill_nulls', Imputer()),
            ('select', RFE(XGBClassifier(n_estimators=100), step=1)),
            ('classifier', XGBClassifier(n_estimators=100))
        ]
    )
    # grid_search = GridSearchCV(pipeline, xgbclassifier_constants.param_grid, scoring='roc_auc', verbose=10, n_jobs=-1, cv=5)
    grid_search = RandomizedSearchCV(pipeline, classifier_constants.param_grid_randomized, n_iter=1000, scoring='roc_auc', verbose=10, n_jobs=-1, cv=5)  # n_iter * cv fits will be made here

    grid_search.fit(X, y)
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    joblib.dump(grid_search, 'xgboost_clsf_model.pkl')

def model_validate():
    X = joblib.load('X_test.pkl')
    y = joblib.load('y_test.pkl')
    model = joblib.load('xgboost_clsf_model.pkl')
    print(roc_auc_score(y, model.predict_proba(X)[:, 1]))

if __name__ == '__main__':
    preprocess_raw_data()
    preprocess_objects_create()
    model_train()
    model_validate()
