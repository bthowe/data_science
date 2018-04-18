import sys
import sqlite3
import pandas as pd

pd.set_option('max_columns', 10000)
pd.set_option('max_info_columns', 10000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 40000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


sqlite_file = '/Users/travis.howe/Downloads/world-development-indicators.sqlite'
conn = sqlite3.connect(sqlite_file)
c = conn.cursor()

# df = pd.read_sql_query('SELECT * FROM sqlite_master;', conn)
# print(df.head())

# df = pd.read_sql_query('SELECT * FROM Country;', conn)
# df = pd.read_sql_query('SELECT * FROM CountryNotes;', conn)
# df = pd.read_sql_query('SELECT * FROM Series;', conn)
df = pd.read_sql_query('SELECT * FROM Indicators WHERE CountryName in ("Bermuda", "Sri Lanka");', conn)

df_wide = df.pivot_table(values='Value', index=['CountryName', 'Year'], columns='IndicatorName').reset_index()
# print(df_wide['Life expectancy at birth, total (years)'].head())
# print(df_wide['Life expectancy at birth, total (years)'].describe())
print(df_wide)

# CountryName
# CountryCode
# IndicatorName
# IndicatorCode
# Year
# Value
conn.close()


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

# TODO: (1) What are the regions I care about? Specific countries, what? (2) What would I do with the covariates if I were unconstrained?, (3) How to actually restrict variables (how much does each vary?)
# (4) Need to deal with missing values