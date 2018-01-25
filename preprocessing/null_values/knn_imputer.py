import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# todo: make into a class



def rf_similarity(fitted_estimator, x, X):
    """
    estimator: tree-based, fitted estimator
    x: the point of comparison
    X_other: the data to compare to x

    Returns the indeces of the most similar observations from X_other.
    """
    x_leafs = fitted_estimator.apply(x)
    X_leafs = fitted_estimator.apply(X)

    similarity = np.where(x_leafs == X_leafs, 1, 0).sum(axis=1)

    indeces = np.argwhere(similarity == np.max(similarity)).flatten()
    return X.index[indeces].values


def rf_func(df, feature_to_impute, min_samples_leaf):
    """feature_to_impute: name of feature to fill the missing values"""

    df_to_impute = df.query('{0} != {0}'.format(feature_to_impute))

    estimate_cols = [col for col in df.columns.tolist() if col not in [feature_to_impute, 'target']]

    X = df.dropna()[estimate_cols + ['target']]
    y = X.pop('target')

    rf = RandomForestClassifier(n_estimators=5, min_samples_leaf=min_samples_leaf)
    rf.fit(X, y)

    for index in X.index:
        row = pd.DataFrame(X.loc[index]).transpose()
        print(rf_similarity(rf, row, X.drop(index)))

    sys.exit()






    sys.exit()
    print(X)
    print(rf.apply(X))
    a = rf.apply(X)
    indeces = X.index.values.reshape((len(X), 1))


    print(np.argmax(np.where(a == a[0, :], 1, 0).sum(axis=1)))



    for index, row in enumerate(a):
        similarity = np.where(row == a, 1, 0).sum(axis=1)
        similarity[index] = -1

        print(similarity)
        print(np.argwhere(similarity == np.max(similarity)))

        print(index)

        sys.exit()
        a_row = a[i, :]
        a_rest = a

        np.argwhere(list == np.amax(list))




    # todo: go from this to some type of count of how often they are in the same group


    sys.exit()


    rfneighbors = rf.classes_  #.kneighbors(df_to_impute[estimate_cols])[1]

    knn_means = [df.loc[neighbors][estimate_cols].values.mean() for neighbors in kneighbors]

    df_to_impute[feature_to_impute] = knn_means

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

    knn_means = [df.loc[neighbors][estimate_cols].values.mean() for neighbors in kneighbors]

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

    # knn_func(df, 'one', 3)
    rf_func(df, 'one', 3)