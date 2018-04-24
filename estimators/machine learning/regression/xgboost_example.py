import sys
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import uniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def model_train_random():
    X = joblib.load('../data_files/X_train.pkl')
    y = joblib.load('../data_files/y_train_reg.pkl')

    xgb_parameters = {
        'max_depth': [2, 3, 4, 5],
        'learning_rate': uniform(),
        'n_estimators': [10, 50, 100],
        'booster': ['gbtree', 'gblinear', 'dart'],
        'min_child_weight': [0, 1, 2, 3],
        'max_delta_step': [0, 1, 2, 3],
        'colsample_bytree': uniform(),
        'colsample_bylevel': uniform(),
        'reg_lambda': uniform()
    }

    grid_search = RandomizedSearchCV(
        xgb.XGBRegressor(),
        xgb_parameters,
        n_iter=200,
        scoring='neg_mean_absolute_error',
        verbose=10,
        n_jobs=-1,
        cv=5
    )
    grid_search.fit(X, y)
    print(grid_search.best_params_)
    print(grid_search.best_score_)

if __name__ == '__main__':
    model_train_random()
