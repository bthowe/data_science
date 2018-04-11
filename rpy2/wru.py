import os
import sys
import joblib
import datetime
import requests
import numpy as np
import pandas as pd
import censusgeocode as cg

os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

census_key = os.getenv('CENSUS_KEY')

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def feature_create(df):
    df['age'] = (pd.datetime.today() - pd.to_datetime(df['birth_date'])).dt.days / 365.25
    df['sex'] = df['gender'].apply(lambda x: 0 if x == 'Male' else 1)  # coded as 0 for Male and 1 for Female because that is the definition used in wru.predict_race
    return df

def _batch_prep(df):
    df[['street_address', 'city', 'postal_abbreviation', 'code']].\
        dropna().\
        to_csv('/Users/travis.howe/Downloads/test_address2.csv', header=False, index=True)

def geocode_batch(df):
    _batch_prep(df)
    # todo: is there anything I can do to increase the number of matches---about 89% of observations have a match

    df['id'] = df.index
    vars = ['id', 'last_name', 'age', 'sex', 'postal_abbreviation']

    df_census = pd.DataFrame(cg.addressbatch('/Users/travis.howe/Downloads/test_address2.csv', returntype='geographies'))
    df_census['id'] = df_census['id'].astype(int)

    joblib.dump(df[vars].merge(df_census, how='outer', on='id', indicator=True), 'results.pkl')

def predict_prep():
    df = joblib.load('results.pkl')
    covars = ['last_name', 'postal_abbreviation', 'countyfp', 'tract', 'block', 'age', 'sex']
    return df[covars].rename(columns={'last_name': 'surname', 'postal_abbreviation': 'state', 'countyfp': 'county'})

def race_predict(df):
    # todo: why are there missing counties?
    df = df.query('(county != "None") and (county == county)')

    r = robjects.r
    pandas2ri.activate()
    wru = importr('wru')  # https://github.com/kosukeimai/wru

    df.loc[3, 'surname'] = 'Althaus'

    df.dropna(inplace=True)
    df['age'] = df['age'].apply(lambda x: round(x))

    X_out = wru.predict_race(voter_file=df, census_geo='county', census_key=census_key, sex=True, age=True)
    print(pandas2ri.ri2py(X_out))

if __name__ == '__main__':
    df = joblib.load('/Users/travis.howe/Downloads/addresses.pkl')

    feature_create(df).pipe(geocode_batch)

    predict_prep().pipe(race_predict)

# todo: download the census data so don't have to query the website whenever I want to generate a prediction
# todo: there is likely a better way to county data from address...look into this.

# todo: Althaus Jr; changing it to Althaus solved the problem.
# todo: some people don't have ages, gender
# todo: age can't be decimal
# todo: does census_geo='tract', census_geo='block' give a different answer
# todo: what to do if a person doesn't get a race estimate
