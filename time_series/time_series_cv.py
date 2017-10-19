import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

# todo: I'd like to pass in the models and their parameter grids
# todo: maybe just start with one model and a singleton grid
# todo: extend to other metrics?

# can I somehow just bring in the sklearn cv thingo?

class TimeSeriesCrossVal(object):
    def __int__(self, model, param_grid, min_train_obs, lag_to_score):
        self.model = model
        self.param_grid = param_grid
        self.min_train_obs = min_train_obs
        self.lag_to_score = lag_to_score

        self.model_dict = {}
        self.best_model = None

    def _rolling_origin(self, X, y, model):
        ssr = 0
        for i in xrange(self.min_train_obs, len(X) - self.lag_to_score):
            X_train = X.iloc[:i]
            y_train = y.iloc[:i]
            X_test = X.iloc[i + self.lag_to_score, i + self.lag_to_score + 1]
            y_test = y.iloc[i + self.lag_to_score, i + self.lag_to_score + 1]

            model.fit(X_train, y_train)
            ssr += (model.predict(X_test) - y_test) ** 2

        return ssr / len(xrange(self.min_train_obs, len(X) - self.lag_to_score))

    def fit(self, X, y):
        model = clone(self.model)
        m = model(**self.param_grid)
        score = self._rolling_origin(X, y, m)
        self.model_dict[m] = score

        self.best_model = max(self.model_dict, key=self.model_dict.get)

        return self

    def predict(self, X):
        return self.best_model.predict(X)



if __name__ == '__main__':
    size_df = 100

    df = pd.DataFrame()
    df['date'] = np.arange(size_df)
    df['x1'] = np.random.randint(0, 100, size_df)
    df['L1'] = 0
    df['L1'].iloc[0] = 10
    df['outcome'] = 7 + df['x1'] * .3 + df['L1'] * 5 + norm.rvs(loc=0, scale=1, size=size_df)
    for i in xrange(1, size_df):
        df['L1'].iloc[i] = df['outcome'].iloc[i-1]
        df['outcome'].iloc[i] = 7 + df['x1'].iloc[i] * .3 + df['L1'].iloc[i] * .5 + norm.rvs(loc=0, scale=1, size=1)

    df = df.iloc[-10:].reset_index(drop=True)
    y = df.pop('outcome')
    X = df
    #
    # model = Ridge(fit_intercept=True)
    # param_grid = {'alpha': 1}
    # min_train_obs = 5
    # lag_to_score = 1
    # tscv = TimeSeriesCrossVal(model, param_grid, min_train_obs, lag_to_score)
    # tscv.fit(X, y)

    # todo: specify a minimum number
    # todo: specify which to test
    tscv = TimeSeriesSplit(n_splits=9)
    for train_index, test_index in tscv.split(X):
        print "Train:", train_index, "Test:", test_index

#         change
