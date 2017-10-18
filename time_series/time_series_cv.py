from sklearn.base import clone

# todo: I'd like to pass in the models and their parameter grids
# todo: maybe just start with one model and a singleton grid
# todo: extend to other metrics?
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



# todo: test this





