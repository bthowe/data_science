import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

def feature_pull(sd, ed):
    query = '''
        SELECT *
        FROM database.table
        WHERE date(database.table.time) >= {0} AND date(database.table.time) <= {1};
    '''.format(sd, ed)
    return svg.get_data_from_mysql(unique_keyword_query=query, connection=svg.remote_connection(db_config, ssl_args))

def feature_type(df):
    df['is_uw'] = df['is_uw'].apply(lambda x: ord(x))
    df['is_oe'] = df['is_oe'].apply(lambda x: ord(x))
    df['is_gi'] = df['is_gi'].apply(lambda x: ord(x))
    df['is_p'] = df['is_p'].apply(lambda x: 1 if x == b'\x01' else 0)

    df['birthdate'] = pd.to_datetime(df['birthdate'])
    return df

def observation_filter(df):
    return df.\
        loc[~((df['is_uw'] + df['is_oe'] + df['is_gi']) > 1)]. \
        query('is_p == 1').\
        query('thingo_date >= "2017-12-21"').\
        drop_duplicates()

def target_create(df):
    df['target'] = 0
    df.loc[df['submission_date'] != df['submission_date'], 'target'] = 1
    return df

def feature_create(df):
    df['ramp_dummy'] = df.apply(lambda x: 1 if x['tenure'] < 4 else 0, axis=1)
    df.loc[:, 'month'] = df['time'].dt.month
    df.loc[:, 'dayofweek'] = df['time'].dt.dayofweek
    return df

def feature_drop(df):
    feature_lst = ['covar1', 'covar2', 'covar3']
    return df[feature_lst]

class Dummies_Create(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.v = DictVectorizer(sparse=False)

    def fit(self, df, y=None, **fit_params):
        self.v.fit(row for _, row in df.iterrows())
        return self

    def transform(self, df, **transform_params):
        return pd.DataFrame(self.v.transform(row for _, row in df.iterrows()), columns=self.v.feature_names_)
