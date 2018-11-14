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
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import uniform

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def model_train(X):
    y = X.pop('active')
    joblib.dump(X, 'X_train.pkl')
    joblib.dump(y, 'y_train.pkl')

    pipeline = Pipeline(
        [
            ('feature_drop', FunctionTransformer(helpers.feature_drop, validate=False)),
            ('feature_dummies', helpers.Dummies_Create()),
            ('xgb', XGBClassifier(n_estimators=100))
        ]
    )

    xgb_grid_params = {
        'xgb__max_depth': [3, 4, 5, 6, 7],
        'xgb__learning_rate': [0.1, .5, 1],
        'xgb__reg_lambda': [0.1, .5, 1],
        'xgb__min_child_weight': [10, 12, 14, 16, 18]
    }
    xgb_rand_params = {
        'xgb__max_depth': [2, 3, 4, 5, 6],
        'xgb__learning_rate': uniform(),
        'xgb__reg_lambda': uniform(),
        'xgb__min_child_weight': [4, 6, 8, 10, 12]
    }

    grid_search = RandomizedSearchCV(pipeline, xgb_rand_params, scoring='roc_auc', verbose=10, n_jobs=6, cv=3, n_iter=100)
    grid_search.fit(X, y)
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    joblib.dump(grid_search, 'grid_search.pkl')

def train(start_date, end_date):
    helpers.data_pull().\
        pipe(helpers.feature_create).\
        pipe(helpers.observation_filter_train, start_date, end_date).\
        pipe(helpers.target_create_train).\
        pipe(model_train)

def test(end_date):
    model = joblib.load('grid_search.pkl')
    X = helpers.data_pull().\
        pipe(helpers.feature_create). \
        pipe(helpers.observation_filter, end_date). \
        pipe(helpers.target_create)

    y = X.pop('active')

    print(pd.Series(model.predict(X)).describe())

def day_stats(end_date):
    model = joblib.load('grid_search.pkl')
    X = helpers.data_pull(). \
        pipe(helpers.feature_create). \
        pipe(helpers.observation_filter, end_date)
    X['active_pred'] = model.predict(X)
    X.query('active_pred == 1', inplace=True)

    def daily_stuff(x):
        x_time_calc = x.query('((button == "Submit") and (page == "Main")) or (button == "Main Menu") or (button == "Quit")')
        x_time_calc['Lpage'] = x_time_calc['page'].shift(1)
        x_time_calc['Ltimestamp'] = x_time_calc['timestamp'].shift(1)
        x_time_calc['time_diff'] = x_time_calc['timestamp'] - x_time_calc['Ltimestamp']

        x_time = x_time_calc.iloc[range(1, len(x_time_calc) + 1, 2)]  # can I groupby just the values in a list?
        x_time = x_time['time_diff'].groupby(x_time['page']).sum().reset_index()

        x['total_work_time'] = x['timestamp'].iloc[-1] - x['timestamp'].iloc[0]
        if len(x_time.query('page == "Practice"')) > 0:
            x['practice_time'] = x_time.query('page == "Practice"')['time_diff'].iloc[0]
        else:
            x['practice_time'] = datetime.timedelta(seconds=0)
        if len(x_time.query('page == "Quiz"')) > 0:
            x['quiz_time'] = x_time.query('page == "Quiz"')['time_diff'].iloc[0]
        else:
            x['quiz_time'] = datetime.timedelta(seconds=0)

        x['practice_num'] = x.query('page == "Practice"').query('button != Lbutton').query('button == "Flip"').shape[0]
        x['quiz_num'] = x.query('page == "Quiz"').query('button != Lbutton').query('button == "Submit"').shape[0]
        return x

    X_g = X.groupby(X['date']).apply(daily_stuff)
    X_new = X_g[['total_work_time', 'practice_time', 'quiz_time', 'practice_num', 'quiz_num']].groupby(X['date']).agg(lambda x: x.value_counts().index[0])
    X_new['average_practice_time_per_card'] = X_new['practice_time'] / X_new['practice_num']
    X_new['average_quiz_time_per_card'] = X_new.apply(lambda x: x['quiz_time'] / x['quiz_num'] if x['quiz_num'] > 0 else np.nan, axis=1)
    print(X_new.info())
    print(X_new)

# todo: (1) total time, (2) number of practice cards and quiz cards, (3) average time per practice card and quiz card
# todo: plot: total time and split into colors by Practice and Quiz
# todo: plot: right on top put the number of cards of each Practice and Quiz


def main():
    start_date = '2018-09-01'
    end_date = '2018-11-09'

    # train(start_date, end_date)
    # test(end_date)
    day_stats(end_date)


if __name__ == '__main__':
    main()

# todo: work on intergrating a neural network
#


