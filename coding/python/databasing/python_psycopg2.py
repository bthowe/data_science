# use py3 environment
import os
import sys
import psycopg2
import pandas as pd
from collections import Counter

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

query = """
        SELECT *
        FROM bacchanalytics.lima_cadence_input
        """

DATABASE = 'datawarehouse'
USER = os.getenv('REDSHIFT_USER')
PASSWORD = os.getenv('REDSHIFT_PASSWORD')
HOST = os.getenv('REDSHIFT_HOST')
PORT = '5439'

def db_connection():
    conn = psycopg2.connect(
        host=HOST,
        port=PORT,
        user=USER,
        password=PASSWORD,
        database=DATABASE,
    )
    return conn

conn = db_connection()

df = pd.read_sql(query, conn).query('yhad_id != ""')
df.dropna(subset=['yhad_id'], inplace=True)

conn.close()

lst = df['yhad_id'].apply(lambda x: x[-1]).values.tolist()

print(Counter(lst))
