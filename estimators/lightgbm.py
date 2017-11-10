import pandas as pd
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, GroupShuffleSplit

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
pipeline = Pipeline(
    [
        ('select', RFE(Lasso(fit_intercept=True, normalize=True), step=1)),
        ('lgb', lgb.LGBMRegressor(objective='regression', n_estimators=500))
    ]
)
lgb_parameters = {
    'lgb__max_depth': [-1],
    'lgb__learning_rate': [0.1],
    'lgb__reg_lambda': [1],
    'lgb__min_child_weight': [.001],
    'lgb__colsample_bytree': [1, .5, .25]
}
grid_search = GridSearchCV(pipeline, lgb_parameters, scoring='neg_mean_absolute_error', verbose=10, n_jobs=-6, cv=3)
grid_search.fit(X_train, y_train)
print grid_search.best_params_
print grid_search.best_score_
print mean_absolute_error(y_test, grid_search.predict(X_test))

# todo: lower level api
