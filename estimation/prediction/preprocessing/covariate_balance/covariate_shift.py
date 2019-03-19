"""Most supervised machine learning techniques are built on the assumption that data at the training and production
stages follow the same distribution. The following tests whether this is true."""


import numpy as np
import pandas as pd
from scipy.stats import uniform
from sklearn.metrics import matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV



def data_create():
    n_obs = 10000
    n_covars = 4

    X_train = pd.DataFrame(np.random.uniform(0, 1, size=(n_obs, n_covars)), columns=list(map(str, range(n_covars))))
    X_train['origin'] = 'train'

    X_production1 = pd.DataFrame(np.random.uniform(0, 1, size=(n_obs, n_covars)), columns=list(map(str, range(n_covars))))
    X_production1['origin'] = 'production1'

    X_production2 = pd.DataFrame(np.random.uniform(-3.5, 4.5, size=(n_obs, n_covars)), columns=list(map(str, range(n_covars))))
    X_production2['origin'] = 'production2'

    return X_train, X_production1, X_production2


def train_test_create(df_train, df_prod, prod_name):
    X_t_train, X_t_test = train_test_split(df_train, test_size=.2)
    X_p_train, X_p_test = train_test_split(df_prod, test_size=.2)

    X_train = X_t_train.append(X_p_train).pipe(pd.get_dummies).drop('origin_{}'.format(prod_name), 1)
    y_train = X_train.pop('origin_train')

    X_test = X_t_test.append(X_p_test).pipe(pd.get_dummies).drop('origin_{}'.format(prod_name), 1)
    y_test = X_test.pop('origin_train')

    return X_train, X_test, y_train, y_test


def model_create(X, y):
    param_grid_randomized = {
        'penalty': ['l1', 'l2'],
        'C': uniform(0, 10),
        'fit_intercept': [True, False]
    }
    grid_search = RandomizedSearchCV(LogisticRegression(), param_grid_randomized, n_iter=1000, scoring='roc_auc', verbose=10, n_jobs=-1, cv=5)  # n_iter * cv fits will be made here
    grid_search.fit(X, y)
    return grid_search


def evaluate_model(model, X, y_actual):
    y_pred = model.predict(X)
    mcc = matthews_corrcoef(y_actual, y_pred)
    print('Value of the Matthews Correlation Coefficient: {}.'.format(mcc))
    if mcc > .2:
        print('There appears to have been a shift in the covariates.')
    else:
        print('There does not appear to have been a shift in the covariates.')


def main():
    X_t, X_prod1, X_prod2 = data_create()
    X_train, X_test, y_train, y_test = train_test_create(X_t, X_prod2, 'production2')
    model = model_create(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == '__main__':
    main()

    # todo: This should be done a number of time and the results averaged.
    # todo: This method detects a shift if the lower and upper bound of the covariates are changed from (0, 1) to
    # (0.5, 1.5). However, no change is detected if the mean remains the same and the variance increases, to
    # (-3.5, 4.5), say. This is due to the choice of classifier.
