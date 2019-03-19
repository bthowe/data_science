import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def rf_similarity(fitted_estimator, x, X):
    """
    estimator: tree-based, fitted estimator
    x: the point of comparison
    X_other: the data to compare to x

    Returns a count for each data point in X of the number of trees in which it shares a leaf node with x
    """
    x_leafs = fitted_estimator.apply(x)
    X_leafs = fitted_estimator.apply(X)

    similarity = np.where(x_leafs == X_leafs, 1, 0).sum(axis=1)
    return dict(zip(X.index, similarity))

class TreeImpute(object):
    # todo: if multiple features in the data point are null then it breaks. How should I proceed here? I'd have to use those columns that aren't null.
    """
    estimator: the unfitted, tree-based estimator object
    impute_feature: name of the feature in the data with missing values to fill
    model_features: names of the features in the data to use to train the model used to fill the null values

    X: full (i.e., includes missing values) dataframe of covariates
    y: corresponding series of outcome variables

    Returns df with imputed values in place of the previous null values in feature_to_impute.
    """

    def __init__(self, estimator, impute_feature, model_features):
        self.estimator = estimator
        self.impute_feature = impute_feature
        self.model_features = model_features

    def impute(self, X, y):
        df_to_impute = X.query('{0} != {0}'.format(self.impute_feature))
        X_analysis = X.query('{0} == {0}'.format(self.impute_feature))
        y_analysis = y.loc[X.index]

        X_train, X_test, y_train, y_test = train_test_split(X_analysis, y_analysis, test_size=0.33, random_state=42)

        self._fit(X_train, y_train)

        X.loc[df_to_impute.index, self.impute_feature] = self._mean_calc(df_to_impute, X_test)
        return X

    def _fit(self, X, y):
        X = X[self.model_features].dropna()
        y = y.loc[X.index]
        self.estimator.fit(X, y)

    def _mean_calc(self, df_to_impute, X_test):
        rf_means = []
        for index in df_to_impute.index:
            row = pd.DataFrame(df_to_impute.loc[index]).transpose()[self.model_features]
            count_dic = rf_similarity(self.estimator, row, X_test[self.model_features].dropna())
            neighbors = list(filter(lambda x: count_dic[x] == max(count_dic.values()), count_dic.keys()))
            rf_means.append(df.loc[neighbors][self.impute_feature].values.mean(axis=0))
        return rf_means

if __name__ == '__main__':
    np.random.seed(4)
    df = pd.concat(
        [
            pd.DataFrame(np.random.uniform(0, 1, size=(20, 3)), columns=['one', 'two', 'three']),
            pd.DataFrame(np.random.randint(0, 2, size=(20, 1)), columns=['target'])
        ], axis=1
    )
    df.loc[np.random.choice(20, size=2), 'one'] = np.nan
    df.loc[np.random.choice(20, size=2), 'two'] = np.nan

    X = df
    y = df.pop('target')
    impute_feature = 'one'
    model_features = ['two', 'three']

    print(X)
    ti = TreeImpute(RandomForestClassifier(n_estimators=5, min_samples_leaf=2), impute_feature, model_features)
    print(ti.impute(X, y))
