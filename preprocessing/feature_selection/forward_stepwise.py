import sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

class ForwardStepwiseSelector(object):
    """
    This class performs feature selection by iteratively adding features to the model based on which increases the score
    by most in each step. If no feature increases the score, then none are added to the model.
    """
    # todo: more testing to make sure there isn't a mistake
    # todo: pass scoring function in
    # todo: step size?
    def __init__(self, estimator):
        self.estimator = estimator

        self.feature_lst = []
        self.model_feature_lst= []

        self.new_score = None
        self.best_score = 0

    def fit(self, X, y):
        self.feature_lst = X.columns.tolist()
        self._stepwise(X, y)
        return self

    def _stepwise(self, X, y):
        feature_to_add, new_score = self._cv_iterate()

        if new_score > self.best_score:
            self.model_feature_lst.append(feature_to_add)
            self.feature_lst.remove(feature_to_add)
            self.best_score = new_score
            self._stepwise(X, y)

    def _cv_iterate(self):
        feature_score_dict = {}
        for feature in self.feature_lst:
            feature_score_dict[feature] = cross_val_score(self.estimator, X[self.model_feature_lst + [feature]], y, scoring='roc_auc').mean()
        best_feature = max(feature_score_dict, key=feature_score_dict.get)
        best_score = feature_score_dict[best_feature]
        return (best_feature, best_score)

    def transform(self, X, **transform_params):
        model_feature_lst_ordered = [feature for feature in X.columns.tolist() if feature in self.model_feature_lst]
        return X[model_feature_lst_ordered]

if __name__ == '__main__':
    N = 1000
    np.random.seed(2)
    columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']
    df = pd.DataFrame(np.random.uniform(-1, 1, size=(N, len(columns))), columns=columns)
    df['target'] = np.random.randint(0, 2, size=(N, 1))

    X = df
    y = X.pop('target')

    fs = ForwardStepwiseSelector(XGBClassifier())
    print(fs.fit(X, y).transform(X).head())
