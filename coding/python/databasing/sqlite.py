import sys
import joblib
import sqlite3
import pandas as pd

pd.set_option('max_columns', 10000)
pd.set_option('max_info_columns', 10000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 40000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def data_create():
    sqlite_file = '/Users/travis.howe/Downloads/world-development-indicators.sqlite'
    conn = sqlite3.connect(sqlite_file)
    c = conn.cursor()

    # df = pd.read_sql_query('SELECT * FROM sqlite_master;', conn)
    # print(df.head())

    # df = pd.read_sql_query('SELECT * FROM Country;', conn)
    # df = pd.read_sql_query('SELECT * FROM CountryNotes;', conn)
    # df = pd.read_sql_query('SELECT * FROM Series;', conn)
    df = pd.read_sql_query('SELECT * FROM Indicators;', conn).iloc[3492:]

    # print(df.iloc[3492:].tail(500))
    # sys.exit()

    df_wide = df.pivot_table(values='Value', index=['CountryName', 'Year'], columns='IndicatorName').reset_index()
    # print(df_wide['Life expectancy at birth, total (years)'].head())
    # print(df_wide['Life expectancy at birth, total (years)'].describe())

    # CountryName
    # CountryCode
    # IndicatorName
    # IndicatorCode
    # Year
    # Value
    conn.close()
    return df_wide


    # Population, total
    # Mortality rate, adult, male (per 1,000 male adults)
    # Mortality rate, adult, female (per 1,000 male adults)
    # Death rate, crude (per 1,000 people)
    # Life expectancy at birth, female (years)
    # Life expectancy at birth, male (years)
    # Survival to age 65, female (% of cohort)
    # Mortality rate, under-5 (per 1,000)
    # Mortality rate, infant (per 1,000 live births)

    # target
    # Life expectancy at birth, total (years)


def feature_choose(df):
    # print(df.info())
    # print(df['CountryName'].unique())

    df.query('CountryName == "United States"', inplace=True)
    print(df.head())
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    df = df.set_index('Year')
    print(df.info())
    print(df.head())

    # print(df.interpolate(method='time'))







if __name__ == '__main__':
    # joblib.dump(data_create(), '/Users/travis.howe/Downloads/raw_data.pkl')
    joblib.load('/Users/travis.howe/Downloads/raw_data.pkl').pipe(feature_choose)



# TODO: (1) What are the regions I care about? Specific countries, what? (2) What would I do with the covariates if I were unconstrained?, (3) How to actually restrict variables (how much does each vary?)
# (4) Need to deal with missing values

# Make some decision about the regions
# Use imputer
# linear regression

# if I had more resources
# 1. different ways of filling the nulls
#   a. random forest
#   b. nearest neighbors
#   c. dropping, if missing at random
#   d. using a mean or median value
#   e. interpolation and extrapolation (try this one)
# 2. dealing with the many features
#   a. RFE
#   b. feature addition
#   c. PCA
#   d. Lasso
#   e. straight OLS and only keep significant values (do this one)
#   f. variance threshold (problematic if use interpolation)
#   g. just drop features if too sparse
# 3. feature creation
#   a. interactions
#   b. other data sets?
# 4. models (deal with hierarchical nature and time-dependencies)
#   a. fixed effects
#   b. split by group (i.e., groups need to be left together)
#   c. could add autoregressive term
#       a. could be problematic if missing a lot of outcome instances
#   d. add a country specific linear trend
# 5. dealing with missing outcome variables
#   a. do not interpolate or extrapolate
#   b. do nothing...simply rely on the linear trend.
