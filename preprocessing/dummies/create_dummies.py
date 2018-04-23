import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

class Dummies_Create(BaseEstimator, TransformerMixin):
    """This class creates dummy variables in a dataset. The fit method creates the dictvectorizer object, and the
    transform method uses it to transform the dataset according to the initial data. """

    def __init__(self):
        self.v = DictVectorizer(sparse=False)

    def fit(self, df, y=None, **fit_params):
        self.v.fit(row for _, row in df.iterrows())
        return self

    def transform(self, df, **transform_params):
        return pd.DataFrame(self.v.transform(row for _, row in df.iterrows()), columns=self.v.feature_names_)
