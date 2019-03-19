import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression


# todo: should I calculate both types (i.e., discrete and continuous) using pseudo-R-squared?
# todo: could make an option since I could estimate the linear model with OLS and binary outcome variable
def _mcfaden_pseudo_r_2(y_true, y_pred_full, y_pred_basic):
    return 1 - (log_loss(y_true, y_pred_full) / log_loss(y_true, y_pred_basic))

def vif_pseudo(X):
    """
    Calculates the variable inflation factor (VIF) for each covariate.

    :param X: Dataframe of covariates (i.e., features)
    :return: The VIF for each variable.
    """
    vif_dict = {}
    for column in X:
        X_temp = X.copy()
        y_temp = X_temp.pop(column)

        if np.all(y_temp.unique() == [0, 1]) or np.all(y_temp.unique() == [1, 0]):  # if a dummy
            lr = LogisticRegression(fit_intercept=False)
            X_temp['constant'] = 1

            r2 = _mcfaden_pseudo_r_2(
                y_temp,
                lr.fit(X_temp, y_temp).predict(X_temp),
                lr.fit(X_temp[['constant']], y_temp).predict(X_temp[['constant']])
            )
        else:
            lr = LinearRegression()

            r2 = r2_score(y_temp, lr.fit(X_temp, y_temp).predict(X_temp))

        vif_dict[column] = 1 / (1 - r2)

    return vif_dict

if __name__ == '__main__':
    N = 1000
    np.random.seed(2)
    df = pd.DataFrame(np.random.uniform(-1, 1, size=(N, 3)), columns=['X1', 'X2', 'X3'])
    df['X4'] = np.random.randint(0, 2, size=(N, 1))

    df['X5'] = df['X1'] + np.random.normal(0, .1, size=(N,))
    df['X6'] = np.abs(df['X4'] - 1)

    print(vif_pseudo(df))
