import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def ls_coefficients(X, y):
    lr = LinearRegression(fit_intercept=True)
    lr.fit(X, y)

    coefficients = list(lr.coef_)
    coefficients.append(lr.intercept_)

    return dict(zip(X.columns.tolist(), coefficients))

def ls_treatment_effect_estimator(X, y, W):
    coeff_dic = ls_coefficients(X, y)
    return coeff_dic[W]

def V(df, W, Y, coeffs, type='hetero'):
    Y = df.pop(Y)
    W = df.pop(W)
    X = df.values

    N = len(df)
    M = X.shape[1]

    alpha = coeffs[0]
    tau = coeffs[1]
    betas = coeffs[2:]

    if type == 'hetero':
        return (1 / (N * (N - 1 - M))) * np.sum(((W - W.mean()) ** 2) * (Y - alpha - tau - betas * X) ** 2) * (1 / (W.mean() * (1 - W.mean())) ** 2)
    else:
        return (1 / (N * (N - 1 - M))) * np.sum((Y - alpha - tau - betas * X) ** 2) * (1 / (W.mean() * (1 - W.mean())))

