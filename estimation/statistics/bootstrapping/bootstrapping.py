import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

def outcome_condition_distribution_bootstrap(model, X, y):
    """"In the 'linear regression model'
                        y_i = a + bx_i + varepsilon_i.
    The assumptions are No SELFES (Normality, Strict Exogeneity, Linearity, Full rank, Errors are Spherical).
        -strict exogeneity (the errors are not correlated with the X's': selection, omitted variable bias, reverse causality)
        -spherical errors (errors are homoskedastic and no auto-correlation)
        -full rank (the matrix X^T X is invertible)
        -linearity (in terms of the parameters (i.e., the betas)

    The normality assumption can be written : e_i ~ N(0, sigma^2). This means, Y|X ~ N(a + bx, sigma^2). The value of
    sigma^2 is estimated by RSS / (n-2), where n is the number of observations.
    """
    n = len(y)
    mu = model.predict(X)
    SSR = np.sum((y - mu)**2)
    se = np.sqrt(SSR / (n-2))
    return np.random.normal(mu, se, size=(n,))

def error_resampling(X, y):
    model = joblib.load()
    e = y - model.predict(X)
    e_resampled = np.random.choice(e, size=len(y), replace=True)
    return X, pd.Series(model.predict(X) + e_resampled, index=y.index)

def observation_resampling(X, y):
    X_new = X.sample(len(X), replace=True)
    y_new = y.loc[X_new.index]
    return X_new, y_new

class BootStrapper(object):
    def __init__(self, n_samples, sampler):
        self.n_samples = n_samples
        self.sampler = sampler

        self.X = None
        self.y = None
        self.boot_list = None
        self.pipeline = None
        self.count = 0

    def _bootstrapped_model(self):
        self.count += 1
        print 'boostrap iteration {}'.format(self.count)
        X_new, y_new = self.sampler(self.X, self.y)
        pipeline = clone(self.pipeline)
        pipeline.fit(X_new, y_new)
        return pipeline

    def fit(self, X, y, pipeline):
        self.X = X
        self.y = y
        self.pipeline = pipeline
        self.boot_list = [self._bootstrapped_model() for i in xrange(self.n_samples)]
        return self

    def distribution_generate(self, X):
        df = pd.DataFrame(index=X.index)
        for i, model in enumerate(self.boot_list):
            df[i] = model.predict(X)
        return df.transpose()


if __name__ == '__main__':
    # Test these
    model = None
    X = None
    y = None
    outcome_condition_distribution_bootstrap(model, X, y)
    error_resampling(model, X, y)