import sys
import joblib
import datetime
import numpy as np
import pandas as pd
import model_helpers
import model_constants
import SVGAnalytics as svg
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import FunctionTransformer, Imputer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


class TwoStageRegression(object):
    """

    model_class: grid search object
    model_reg: grid search object
    """
    def __init__(self, model_class, model_reg):
        self.model_class = model_class
        self.model_reg = model_reg

    def fit(self, X, y):
        y_binary = y.apply(lambda x: 1 if x > 0 else 0)
        y_pos = y.to_frame().query('ltv > 0')
        X_pos = X.loc[y_pos.index.tolist()]

        self.model_class.fit(X, y_binary)
        self.model_reg.fit(X_pos, y_pos)

        return self

    def transform(self, X):
        pred = pd.DataFrame(np.zeros(shape=(len(X), 1)), columns=['ltv_predictions'])
        pred.index = X.index

        non_zero_indeces = X.iloc[np.where(self.model_class.predict_proba(X)[:, 1] > .5)[0]].index
        pred.loc[non_zero_indeces, 'ltv_predictions'] = self.model_reg.predict(X.loc[non_zero_indeces])

        return pred

if __name__ == '__main__':
    X_train = joblib.load('/X_train.pkl')
    y_train = joblib.load('/y_train.pkl')
    X_test = joblib.load('/X_test.pkl')
    y_test = joblib.load('/y_test.pkl')

    # stage 1:
    pipeline = Pipeline(
        [
            ('feature_create', FunctionTransformer(model_helpers.feature_create, validate=False)),
            ('feature_drop', FunctionTransformer(model_helpers.feature_drop, validate=False)),
            ('feature_dummy_create', model_helpers.Dummies_Create()),
            ('feature_interactions_create',
             FunctionTransformer(model_helpers.feature_interactions_create, validate=False)),
            ('select', RFE(XGBClassifier(n_estimators=100), step=1)),
            ('xgb', XGBClassifier(n_estimators=500))
        ]
    )
    grid_search_class = RandomizedSearchCV(pipeline, model_constants.xgb_parameters_randomized, n_iter=150, scoring='roc_auc', verbose=10, n_jobs=-1, cv=3)

    # stage 2:
    pipeline = Pipeline(
        [
            ('feature_create', FunctionTransformer(model_helpers.feature_create, validate=False)),
            ('feature_drop', FunctionTransformer(model_helpers.feature_drop, validate=False)),
            ('feature_dummy_create', model_helpers.Dummies_Create()),
            ('feature_interactions_create', FunctionTransformer(model_helpers.feature_interactions_create, validate=False)),
            ('select', RFE(XGBRegressor(n_estimators=100), step=1)),
            ('xgb', XGBRegressor(n_estimators=500))
        ]
    )
    grid_search_reg = RandomizedSearchCV(pipeline, model_constants.xgb_parameters_randomized, n_iter=150, scoring='neg_mean_absolute_error', verbose=10, n_jobs=-1, cv=3)

    tsr = TwoStageRegression(grid_search_class, grid_search_reg)
    tsr.fit(X_train, y_train)
    print(tsr.transform(X_test))

