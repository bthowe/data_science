from scipy.stats import uniform
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

param_grid = [
    {
        'classifier': [XGBClassifier(n_estimators=100, random_state=42)],
        'classifier__max_depth': [3, 4, 5],
        'classifier__learning_rate': [0.001, 0.01, 0.1],
        'classifier__reg_lambda': [.0001, 0.001, 0.01],
        'classifier__min_child_weight': [10, 12, 14],
        'vt__threshold': [0, .1, .2, .3]
    },
    {
        'classifier': [ExtraTreesClassifier(n_estimators=100)],
        'classifier__criterion': ['gini', 'entropy'],
        'classifier__max_features': ['auto', None],
        'classifier__max_depth': [None, 3],
        'classifier__class_weight': [None, 'balanced', 'balanced_subsample'],
        'vt__threshold': [0, .1, .2, .3]
    },
    {
        'classifier': [LogisticRegression()],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__C': [0.001, 0.01, 0.1, 1, 10],
        'classifier__fit_intercept': [True, False],
        'classifier__class_weight': [None, 'balanced'],
        'vt__threshold': [0, .1, .2, .3]
    }
]

param_grid_randomized = {
    'classifier': [XGBClassifier(n_estimators=100)],
    'classifier__max_depth': [3, 4, 5, 6, 7],
    'classifier__learning_rate': uniform(),
    'classifier__reg_lambda': uniform(),
    'classifier__min_child_weight': [10, 12, 14, 16, 18]
}
