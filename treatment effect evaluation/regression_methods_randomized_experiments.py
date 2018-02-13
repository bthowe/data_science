import numpy as np
import pandas as pd




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

