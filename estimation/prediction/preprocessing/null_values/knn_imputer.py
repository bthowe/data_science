import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

class KnnImpute(object):
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

        self.standardize = StandardScaler()

    def impute(self, X, y):
        df_to_impute = X.loc[X[self.impute_feature] != X[self.impute_feature]]
        X_analysis = X.loc[X[self.impute_feature] == X[self.impute_feature]][self.model_features].dropna()
        y_analysis = y.loc[X_analysis.index]

        self._fit(X_analysis, y_analysis)

        return self._fill(df_to_impute, X)

    def _fit(self, X, y):
        self.estimator.fit(self.standardize.fit_transform(X), y)

    def _mean_calc(self, df_to_impute, X):
        df_to_impute_standardized = self.standardize.transform(df_to_impute[self.model_features])
        kneighbors = self.estimator.kneighbors(df_to_impute_standardized)[1]
        X_analysis = X.loc[~X.index.isin(df_to_impute.index)]
        return [X_analysis.iloc[neighbors][self.impute_feature].values.mean() for neighbors in kneighbors]

    def _fill(self, df_to_impute, X):
        X.loc[df_to_impute.index, self.impute_feature] = self._mean_calc(df_to_impute, X)
        return X

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
    estimator = KNeighborsClassifier(3)  # remember with all Knn problems, (1) standardize, (2) use an odd number of neighbors, (3) try weighted votes, and (4) try different distance metrics

    ki = KnnImpute(estimator, impute_feature, model_features)
    print(ki.impute(X, y))

    # estimator = KNeighborsClassifier(3)
    # features_dropped = ['Rapid Problem Solving', 'Integrity']
    # covars = [col for col in X_train.columns if col not in features_dropped]
    # X_test[features_dropped] = np.nan
    # X = X_train.append(X_test)
    # y = y_train.append(y_test)
    # for remove_feature in features_dropped:
    #     ki = KnnImpute(estimator, remove_feature, covars)
    #     X[remove_feature] = ki.impute(X[covars + [remove_feature]], y)[remove_feature]
    # X_test_imputed = X.loc[X_test.index]
    #
    # print('Imputed Model: {}'.format( roc_auc_score(y_test, model.predict_proba(X_test_imputed.drop(xmc.group_key, 1))[:, 1])))
