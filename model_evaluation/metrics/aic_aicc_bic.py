import numpy as np

def _ssr(y_true, y_pred):
    errors = (y_true - y_pred).values
    return np.dot(errors, errors)

def _sigma2(y_true, y_pred):
    nobs = len(y_true)
    return _ssr(y_true, y_pred) / nobs  # standard error

def fit_scorer(y_true, y_pred, k, metric='AIC'):
    """Assuming the model errors are iid and normally distributed. Otherwise, cannot use estimated error variance."""
    n = len(y_true)
    if metric in ['AIC', 'AICc']:
        score = n * np.log(_sigma2(y_true, y_pred)) + 2 * k
        if metric == 'AICc':  # AIC with bias adjustment
            return score + (2 * k ** 2 + 2 * k) / (n - k - 1)
        return score
    elif metric == 'BIC':
        return n * np.log(_sigma2(y_true, y_pred)) + np.log(n) * k
