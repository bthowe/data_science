from sklearn.base import BaseEstimator, TransformerMixin

class FillNulls(BaseEstimator, TransformerMixin):
    """Fill null values in the dataset passed into the transform method using the values calculated in fit. This is
    useful when gridsearching since it allows you to use values derived from the entire training set and not simply n-1
    folds."""
    def __init__(self):
        self.vals = None

    def fit(self, X, y=None, **fit_params):
        self.vals = X.median()
        return self

    def transform(self, X, **transform_params):
        null_mask = X.isnull().any()
        cols = X.columns.values[null_mask.values]
        for col in cols:
            X[col].fillna(self.vals[col], inplace=True)
        return X
