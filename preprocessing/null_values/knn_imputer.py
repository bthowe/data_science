import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# todo: make into a class



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


def rf_func(df, feature_to_impute, min_samples_leaf):
    # todo: should I just use most similar points or allow user to specify the N most similar points
    """feature_to_impute: name of feature to fill the missing values"""

    df_to_impute = df.query('{0} != {0}'.format(feature_to_impute))

    estimate_cols = [col for col in df.columns.tolist() if col not in [feature_to_impute, 'target']]

    X = df.dropna()[estimate_cols + ['target']]
    y = X.pop('target')

    rf = RandomForestClassifier(n_estimators=5, min_samples_leaf=min_samples_leaf)
    rf.fit(X, y)

    rf_means = []
    for index in X.index:
        row = pd.DataFrame(X.loc[index]).transpose()
        count_dic = rf_similarity(rf, row, X.drop(index))
        neighbors = list(filter(lambda x: count_dic[x] == max(count_dic.values()), count_dic.keys()))
        rf_means.append(df.loc[neighbors][feature_to_impute].values.mean(axis=0))
        # rf_means.append(df.loc[neighbors][estimate_cols].values.mean(axis=0))

    df_to_impute[feature_to_impute] = rf_means

    df.loc[df_to_impute.index] = df_to_impute

    return df


def knn_func(df, feature_to_impute, k):
    # todo: standardize
    # todo: odd number of neighbors
    # todo: weighted votes
    # todo: try different distance metrics

    """feature_to_impute: name of feature to fill the missing values"""

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


if __name__ == '__main__':
    np.random.seed(2)
    df = pd.concat(
        [
            pd.DataFrame(np.random.uniform(0, 1, size=(20, 3)), columns=['one', 'two', 'three']),
            pd.DataFrame(np.random.randint(0, 2, size=(20, 1)), columns=['target'])
        ], axis=1
    )
    df.loc[np.random.choice(20, size=2), 'one'] = np.nan
    df.loc[np.random.choice(20, size=2), 'two'] = np.nan

    knn_func(df, 'one', 2)
    rf_func(df, 'one', 3)