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

def _geocode_batch(df):
    df[['street_address', 'city', 'postal_abbreviation', 'code']].dropna().to_csv('/Users/travis.howe/Downloads/test_address2.csv', header=False, index=True)
    return pd.DataFrame(cg.addressbatch('/Users/travis.howe/Downloads/test_address2.csv', returntype='geographies'))  # I have to send in a .csv file

def predict_prep(df):
    df['id'] = df.index
    census_vars = ['id', 'countyfp', 'tract', 'block']
    obs_vars = ['id', 'last_name', 'age', 'sex', 'postal_abbreviation']

    df_census = _geocode_batch(df)[census_vars]
    df_census['id'] = df_census['id'].astype(int)

    return df[obs_vars].merge(df_census, how='outer', on='id', indicator=True).\
        drop(['id', '_merge'], 1).\
        rename(columns={'last_name': 'surname', 'postal_abbreviation': 'state', 'countyfp': 'county'})

def race_predict(df):
    # todo: why are there missing counties?
    df = df.query('(county != "None") and (county == county)')
    df.set_index([[1]], inplace=True)

    r = robjects.r
    pandas2ri.activate()
    wru = importr('wru')  # https://github.com/kosukeimai/wru

    # df.loc[3, 'surname'] = 'Althaus'

    # df.dropna(inplace=True)
    df['age'] = df['age'].apply(lambda x: round(x))

    census_data = joblib.load('data_files/census_data_all_states_county.pkl')
    X_out = wru.predict_race(voter_file=df, census_geo='county', census_key=census_key, sex=True, age=True, census_data=census_data)
    print(pandas2ri.ri2py(X_out))

    census_data = joblib.load('data_files/census_data_all_states_tract.pkl')
    X_out = wru.predict_race(voter_file=df, census_geo='tract', census_key=census_key, sex=True, age=True, census_data=census_data)
    print(pandas2ri.ri2py(X_out))

#     todo: block?

def _convert_to_pandas(r_lst, level):
    r = robjects.r
    l = r.length(r_lst)
    print(l)

    df = pd.DataFrame()
    for region in range(l):  # includes "DC" in addition to the fifty states
        print(r_lst[region][0])
        print(r_lst[region][1])
        print(r_lst[region][2])
        print(r_lst[region][3])
        df = df.append(pandas2ri.ri2py(r_lst[region][3]))
    return df


def download_census_data(level):
    r = robjects.r
    pandas2ri.activate()
    wru = importr('wru')

    states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA",
              "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
              "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
              "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
              "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

    X_out = wru.get_census_data(census_key, states, age=True, sex=True, census_geo=level)

    joblib.dump(X_out, 'census_data_all_states_{}.pkl'.format(level))
    joblib.dump(_convert_to_pandas(X_out, level), 'census_data_all_states_{}_pd.pkl'.format(level))


def main():

    # joblib.dump(_convert_to_pandas(joblib.load('census_data_all_states_county.pkl'), 'county'), 'census_data_all_states_{}_pd.pkl'.format(level))
    # sys.exit()

    # download_census_data('county')
    download_census_data('tract')
    download_census_data('block')



if __name__ == '__main__':
    # main()
    # sys.exit()

    #example
    # pd.DataFrame([['Howe', '809 Logan Ave', 'Belton', 'MO', '64012', 35, 0]], columns=['last_name', 'street_address', 'city', 'postal_abbreviation', 'code', 'age', 'sex']). \
    pd.DataFrame([['Young', '6700 W 138th Ter', 'Overland Park', 'KS', '66223', 26, 0]], columns=['last_name', 'street_address', 'city', 'postal_abbreviation', 'code', 'age', 'sex']). \
        pipe(predict_prep). \
        pipe(race_predict)



# todo: download the census data so don't have to query the website whenever I want to generate a prediction
# todo: there is likely a better way to county data from address...look into this.

# todo: Althaus Jr; changing it to Althaus solved the problem.
# todo: some people don't have ages, gender
# todo: age can't be decimal
# todo: does census_geo='tract', census_geo='block' give a different answer
# todo: what to do if a person doesn't get a race estimate

# todo: is there anything I can do to increase the number of matches---about 89% of observations have a match

