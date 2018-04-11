import os
import sys
import joblib
import datetime
import requests
import numpy as np
import pandas as pd
from us import states
import censusgeocode as cg

os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def test():
    r = robjects.r
    pandas2ri.activate()
    wru = importr('wru')

    # data = load_iris()
    # X = pd.DataFrame(data.data, columns=data.feature_names)
    # y = data.target

    r['load']('/Users/travis.howe/Downloads/voters.RData')
    print(r['voters'])


    # X = r['voters']
    # print(X.append(pd.DataFrame([['11', 'Howe', 'MO', '5', '095', '000000', '0000', '0', '35', '0', 'Ind', '0', '0000']], columns=X.columns.tolist())))
    #
    # print(dir(wru))  #.predict_race(voter.file = X, census.geo = "county", census.key = "5749053aae31684fca0ad8057364a4239c4618e4", party = "PID")
    # X_out = wru.predict_race(voter_file=X, census_geo='county', census_key='5749053aae31684fca0ad8057364a4239c4618e4', party='PID')
    # print(pandas2ri.ri2py(X_out))
    #


# todo: use the API to convert the address to county, tract, block, etc.
# todo: expand the variables conditioned on.



# https://github.com/kosukeimai/wru




# def

    print(states.MD.fips)



def feature_create(df):
    df['age'] = (pd.datetime.today() - pd.to_datetime(df['birth_date'])).dt.days / 365.25
    df['sex'] = df['gender'].apply(lambda x: 0 if x == 'Male' else 1)  # coded as 0 for Male and 1 for Female because that is the definition used in wru.predict_race
    # todo: what to do if the gender is "None"
    return df

def _batch_prep(df):
    df[['street_address', 'city', 'postal_abbreviation', 'code']].to_csv('/Users/travis.howe/Downloads/test_address.csv', header=False, index=True)

def geocode_batch(df):
    vars = ['last_name', 'age', 'sex', 'postal_abbreviation']

    _batch_prep(df)
    #     todo: is there anything I can do to increase the number of matches---about 89% of observations have a match
    df_result = pd.concat(
        [
            df[vars],
            pd.DataFrame(cg.addressbatch('/Users/travis.howe/Downloads/test_address.csv', returntype='geographies'))
        ],
        axis=1
    )
    joblib.dump(df_result, 'results.pkl')

def predict_prep():
    df = joblib.load('results.pkl')
    covars = ['last_name', 'postal_abbreviation', 'countyfp', 'tract', 'block', 'age', 'sex']
    return df[covars].rename(columns={'last_name': 'surname', 'postal_abbreviation': 'state', 'countyfp': 'county'})

def race_predict(df):
    df.dropna(inplace=True)
    print(df.head())
    # sys.exit()


    r = robjects.r
    pandas2ri.activate()
    wru = importr('wru')

    dfr = pandas2ri.py2ri(df.head())
    print(dfr)

    r['load']('/Users/travis.howe/Downloads/voters.RData')
    print(r['voters'])

    X = r['voters']
    print(X.append(pd.DataFrame([['11', 'Howe', 'MO', '5', '095', '000000', '0000', '0', '35', '0', 'Ind', '0', '0000']], columns=X.columns.tolist())))
    print(X[['surname', 'state', 'county', 'tract', 'block']])

    df.loc[3, 'surname'] = 'Althaus'
    df = df.head().drop(['age', 'sex'], 1).iloc[3:4]
    # print(df); sys.exit()
    # rows indexed 2 and 3 are problematic
    # row 2 breaks...the problem was the name: Althaus Jr; changing it to Althaus solved the problem.
    # row 3 return nulls...no idea...need to dig deeper.
    print(df)
    # todo: does census_geo='tract', census_geo='block' give a different answer
    # X_out = wru.predict_race(voter_file=X[['surname', 'state', 'county', 'tract', 'block']], census_geo='county', census_key='5749053aae31684fca0ad8057364a4239c4618e4')  #, party=True)  #, sex=True)
    X_out = wru.predict_race(voter_file=df, census_geo='county', census_key='5749053aae31684fca0ad8057364a4239c4618e4')  #, age=True, sex=True)
    print(pandas2ri.ri2py(X_out))

if __name__ == '__main__':
    test()
    # geocode()
    # geocode_batch()

    # df = joblib.load('/Users/travis.howe/Downloads/addresses.pkl')
    # feature_create(df).pipe(geocode_batch)

    predict_prep().pipe(race_predict)