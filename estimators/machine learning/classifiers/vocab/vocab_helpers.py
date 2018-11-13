import sys
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

def target_create(df):
    df['active'] = 1

    df.at[list(range(217, 224)), 'active'] = 0
    df.at[505, 'active'] = 0
    df.at[506, 'active'] = 0
    df.at[1427, 'active'] = 0
    df.at[1921, 'active'] = 0
    df.at[1922, 'active'] = 0
    df.at[3504, 'active'] = 0
    df.at[3505, 'active'] = 0
    df.at[4069, 'active'] = 0
    df.at[4070, 'active'] = 0
    df.at[6682, 'active'] = 0
    df.at[6683, 'active'] = 0
    df.at[10106, 'active'] = 0
    df.at[list(range(11296, 11302)), 'active'] = 0
    df.at[11862, 'active'] = 0
    return df.reset_index(drop=False)


def feature_create(df):
    df['date'] = df['timestamp'].dt.date  #.astype(str)
    df['hour_of_day'] = df['timestamp'].dt.hour.astype(str)
    df['day_of_week'] = df['timestamp'].dt.dayofweek.astype(str)

    df['Ltimestamp'] = df['timestamp'].shift(1)
    df['time_since_last_button'] = (df['timestamp'] - df['Ltimestamp']).dt.seconds
    df.at[0, 'time_since_last_button'] = 0

    def day_features(x):
        x['new_day'] = 0
        x['new_day'].iloc[0] = 1

        x['Lpage'] = x['page'].shift(1)
        x.at[x['Lpage'] != x['Lpage'], 'Lpage'] = 'None'

        x['Lbutton'] = x['button'].shift(1)
        x.at[x['Lbutton'] != x['Lbutton'], 'Lbutton'] = 'None'

        x['Lhour_of_day'] = x['hour_of_day'].shift(1)
        x.at[x['Lhour_of_day'] != x['Lhour_of_day'], 'Lhour_of_day'] = 'None'

        x['Lday_of_week'] = x['day_of_week'].shift(1)
        x.at[x['Lday_of_week'] != x['Lday_of_week'], 'Lday_of_week'] = 'None'
        return x
    return df.groupby(df['date']).apply(day_features).reset_index(drop=True)


def feature_none(df):
    df.at[df['page'] != df['page'], 'page'] = 'None'
    df.at[df['button'] != df['button'], 'button'] = 'None'
    df.at[df['hour_of_day'] != df['hour_of_day'], 'hour_of_day'] = 'None'
    df.at[df['day_of_week'] != df['day_of_week'], 'day_of_week'] = 'None'
    return df


def feature_drop(df):
    keep = ['button', 'page', 'hour_of_day', 'day_of_week', 'Lpage', 'Lbutton', 'time_since_last_button', 'Lhour_of_day', 'Lday_of_week', 'new_day']
    return df[keep]

class Dummies_Create(BaseEstimator, TransformerMixin):
    """This class creates dummy variables in a dataset. The fit method creates the dictvectorizer object, and the
    transform method uses it to transform the dataset according to the initial data. """

    def __init__(self):
        self.v = DictVectorizer(sparse=False)

    def transform(self, df, **transform_params):
        return pd.DataFrame(self.v.transform(row for _, row in df.iterrows()), columns=self.v.feature_names_)

        # df_to_dict = df.to_dict('records')
        # return pd.DataFrame(self.v.transform(df_to_dict), columns=self.v.feature_names_)

    def fit(self, df, y=None, **fit_params):
        self.v.fit(row for _, row in df.iterrows())
        return self


# todo: Lbutton etc. needs to be dropped

