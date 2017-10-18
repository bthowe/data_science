import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.arima_process as sm
from scipy.stats import norm
from sklearn import linear_model

class AutoRegressiveRegularization:
    """Estimates a linear model with p autoregressive terms and regularization"""

    def __init__(self, order, penalty='l2'):
        self.order = order

        if penalty == 'l1':
            self.model = linear_model.Lasso(alpha=1, fit_intercept=True, normalize=True)
        else:
            self.model = linear_model.Ridge(alpha=1, fit_intercept=True, normalize=True)

        self.cols = []
        self.y_forecast = None
        self.forecast_dates = None
        self.sigma2 = None
        self.ar_coefs = None

    def fit(self, X, y):
        y.set_index('date', drop=True, inplace=True)
        X.set_index('date', drop=True, inplace=True)
        self.cols = X.columns.tolist()

        for lag in xrange(1, self.order + 1):
            X = pd.concat([X, y.shift(lag)], axis=1)
            self.cols.append('L{}'.format(lag))

        X.columns = self.cols

        X.dropna(axis=0, inplace=True)
        y = y.iloc[self.order:]
        self.model.fit(X, y)
        self.ar_coefs = self.model.coef_[0][-self.order:]

        self.y_forecast = y.values[-self.order:]
        self._sigma2(y, X)
        return self

    def predict(self, X):
        self.forecast_dates = X.pop('date')
        for i in xrange(len(X)):
            X_row = X.iloc[i:i + 1]

            for lag in xrange(1, self.order + 1):
                X_row['L{}'.format(lag)] = self.y_forecast[-lag]

            self.y_forecast = np.append(self.y_forecast, self.model.predict(X_row)[0])

        df = pd.concat([pd.Series(self.forecast_dates), pd.Series(self.y_forecast[self.order:])], axis=1)
        df.set_index('date', drop=True, inplace=True)
        df.columns = ['cpa_forecast']
        return df

    def _sigma2(self, y, X):
        errors = (y['target'].values - self.model.predict(X).flatten())
        ssr = np.dot(errors, errors)
        nobs = len(X)
        self.sigma2 = ssr / nobs  # standard error
        return self

    def predict_ci(self, alpha):
        ma_rep = sm.arma2ma([1] + self.ar_coefs.tolist(), [1], nobs=(len(self.y_forecast[self.order:])))

        fcasterr = np.sqrt(self.sigma2 * np.cumsum(ma_rep ** 2))
        const = norm.ppf(1 - alpha / 2.)

        df = pd.concat([pd.Series(self.forecast_dates), pd.DataFrame(
            np.c_[self.y_forecast[self.order:] - const * fcasterr, self.y_forecast[self.order:] + const * fcasterr])],
                       axis=1)
        df.set_index('date', drop=True, inplace=True)
        df.columns = ['forecast_lb', 'forecast_ub']
        return df


def forecast_plot(df, save=False):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df.cpa_forecast, color='r', label='ARMA(2,0) with regularization forecast')
    ax.plot(df.forecast_lb, color='b', label='forecast confidence interval', linestyle='--')
    ax.legend()
    ax.plot(df.forecast_ub, color='b', linestyle='--')
    if save:
        plt.savefig('cpa_forecast_data_files/forecast_ci.png')
    plt.show()


# todo: the coefficient matrix changes if I use the 'l1' norm...make robust


