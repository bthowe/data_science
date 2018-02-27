import sys
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LinearRegression


# def

# seems best to interact the treatment assignment variable with everything.
# then regress, and evaluate the following hypotheses using the test below:



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


def chi2_test(covars, results):
    V = results.cov_params()
    V_t_g = V[covars].loc[covars].values

    params = results.params.loc[covars].values
    return 1 - chi2.cdf(params.T.dot(V_t_g).dot(params), len(params))



if __name__ == '__main__':
    N = 1000
    np.random.seed(2)
    df = pd.DataFrame(np.random.uniform(-1, 1, size=(N, 3)), columns=['X1', 'X2', 'X3'])
    df['W'] = np.random.randint(0, 2, size=(N, 1))
    df['constant'] = 1
    df['varepsilon'] = np.random.normal(size=(N, 1))
    df['Y'] = 0.25 + 0.4 * df['W'] + 0.2 * df['X1'] - 5 * df['X2'] + 10 * df['X3'] + df['varepsilon']

    X = df.drop('varepsilon', 1)
    y = X.pop('Y')

    import statsmodels.api as sm
    model = sm.OLS(y, X)
    results = model.fit()

    covar_lst = ['X1', 'X2', 'X3', 'W']
    print(chi2_test(covar_lst, results))

    # todo: doc string, finesse main block a bit.

    sys.exit()


    print(results.summary())
    print(np.sqrt((1 / (N * (N - 4))) * np.sum((y - results.predict()) ** 2) / (df['W'].mean() * (1- df['W'].mean()))))

    results = model.fit(cov_type='HC3')
    print(results.summary())
    print(np.sqrt((1 / (N * (N - 4))) * np.sum(((df['W'] - df['W'].mean()) ** 2) * ((y - results.predict()) ** 2)) / ((df['W'].mean() * (1- df['W'].mean())) ** 2)))

    # np.sum(((W - W.mean()) ** 2) * (Y - alpha - tau - betas * X) ** 2) * (
    # 1 / (W.mean() * (1 - W.mean())) ** 2)
    # it's the same!

    # what is the hessian here
    # what is the fisher information matrix of the model?