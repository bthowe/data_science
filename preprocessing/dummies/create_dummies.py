import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

class Dummies_Create(BaseEstimator, TransformerMixin):
    """This class creates dummy variables in a dataset. The fit method creates the dictvectorizer object (denoted "v"),
    and the transform method uses it to transform the dataset according to the initial data. The use of the
    DictVectorizer object means the same binary variables will be created when data is transformed as the data used in
     fitting. This is useful, for example, in gridsearching."""

    def __init__(self):
        self.v = None

    def fit(self, df, y=None, **fit_params):
        df_to_dict = df.to_dict('records')
        self.v = DictVectorizer(sparse=False).fit(df_to_dict)
        return self

    def transform(self, df, **transform_params):
        df_to_dict = df.to_dict('records')
        X = pd.DataFrame(self.v.transform(df_to_dict), columns=self.v.feature_names_)
        return X.drop(self._dummy_dropper(), 1)

    def _dummy_dropper(self):
        last = ''
        to_drop = []
        for var in self.v.feature_names_:
            if ('=' in var) and (var.split('=')[0] != last):
                to_drop.append(var)
                last = var.split('=')[0]
        return to_drop
