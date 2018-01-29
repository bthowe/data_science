import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def knn_func(df, feature_to_impute, k):
    # todo: odd number of neighbors
    # todo: weighted votes
    # todo: try different distance metrics

    """feature_to_impute: name of feature to fill the missing values"""

    ss = StandardScaler()
    ss.fit_transform(X, y)
    # inverse_transform


    df_to_impute = df.query('{0} != {0}'.format(feature_to_impute))

    estimate_cols = [col for col in df.columns.tolist() if col not in [feature_to_impute, 'target']]

    X = df.dropna()[estimate_cols + ['target']]
    y = X.pop('target')

    knn = KNeighborsClassifier(k)
    knn.fit(X, y)

    kneighbors = knn.kneighbors(df_to_impute[estimate_cols])[1]

    # todo: which of these is correct?
    knn_means = [df.loc[neighbors][feature_to_impute].values.mean() for neighbors in kneighbors]
    # knn_means = [df.loc[neighbors][estimate_cols].values.mean() for neighbors in kneighbors]

    # todo: something is not right here
    print(knn_means)

    df_to_impute[feature_to_impute] = knn_means

    df.loc[df_to_impute.index] = df_to_impute

    return df



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

    def impute(self, X, y):
        df_to_impute = X.query('{0} != {0}'.format(self.impute_feature))
        X_analysis = X.query('{0} == {0}'.format(self.impute_feature))
        y_analysis = y.loc[X.index]

        # todo: do I want to do a train test split?
        X_train, X_test, y_train, y_test = train_test_split(self._standardize(X_analysis), y_analysis, test_size=0.33, random_state=42)

        self._fit(X_train, y_train)

        X.loc[df_to_impute.index, self.impute_feature] = self._mean_calc(df_to_impute, X_test)

        # todo: unstandardize the data

        return X

    def _standardize(self, X):
        ss = StandardScaler()
        return ss.fit_transform(X)

    def _fit(self, X, y):
        X = X[self.model_features].dropna()
        y = y.loc[X.index]
        self.estimator.fit(X, y)

    def _mean_calc(self, df_to_impute, X_test):
        knn_means = [df.loc[neighbors][feature_to_impute].values.mean() for neighbors in kneighbors]
        # knn_means = [df.loc[neighbors][estimate_cols].values.mean() for neighbors in kneighbors]

        # todo: something is not right here
        print(knn_means)

        df_to_impute[feature_to_impute] = knn_means

        df.loc[df_to_impute.index] = df_to_impute


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
    ki = KnnImpute(KNeighborsClassifier(k), impute_feature, model_features)
    print(ki.impute(X, y))
    sys.exit()

    print(rf_func(df, 'one', 2))
    knn_func(df, 'one', 2)
