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

        self.X_train = None
        self.y_train = None
        self.X_forecast = None
        self.y_forecast = None

        self.forecast_dates = None

        self.sigma2 = None
        self.ar_coefs = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

        y.index = y.date
        y.drop('date', 1, inplace=True)
        X.index = X.date
        X.drop('date', 1, inplace=True)

        self.cols = X.columns.tolist()

        for lag in xrange(1, 2 + 1):
            X = pd.concat([X, y.shift(lag)], axis=1)
            self.cols.append('L{}'.format(lag))

        X.dropna(axis=0, inplace=True)
        y = y.iloc[2:]
        self.model.fit(X, y)

        self.y_forecast = self.y_train.values[-2:]

        self._sigma2()
        return self

    def predict(self, X):
        self.X_forecast = X
        self.forecast_dates = self.X_forecast.pop('date')

        for i in xrange(len(self.X_forecast)):
            X_row = self.X_forecast.iloc[i]

            X_row['Lcpa'] = self.y_forecast[-1]
            X_row['L2cpa'] = self.y_forecast[-2]

            self.y_forecast = np.append(self.y_forecast, self.model.predict(X_row)[0])
        df = pd.concat([pd.Series(self.forecast_dates), pd.Series(self.y_forecast[2:])], axis=1)
        df.set_index('date', drop=True, inplace=True)
        df.columns = ['cpa_forecast']
        return df

    def _sigma2(self):
        errors = (self.y_train[2:] - self.model.predict(self.X_train))['target']
        ssr = np.dot(errors, errors)
        nobs = len(self.X_train)
        self.sigma2 = ssr / nobs  # standard error
        self.ar_coefs = self.model.coef_[0][-2:]
        return self

    def predict_ci(self, alpha):
        ma_rep = sm.arma2ma([1, -self.ar_coefs[0], -self.ar_coefs[1]], [1], nobs=(len(self.X_forecast)))
        fcasterr = np.sqrt(self.sigma2 * np.cumsum(ma_rep ** 2))

        const = norm.ppf(1 - alpha / 2.)

        df = pd.concat([pd.Series(self.forecast_dates), pd.DataFrame(
            np.c_[self.y_forecast[2:] - const * fcasterr, self.y_forecast[2:] + const * fcasterr])],
                       axis=1)
        df.set_index('date', drop=True, inplace=True)
        df.columns = ['forecast_lb', 'forecast_ub']
        return df
