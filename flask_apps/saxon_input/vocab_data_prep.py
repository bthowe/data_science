import datetime
import numpy as np
import pandas as pd
from pymongo import MongoClient
from collections import defaultdict
from flask import Flask, request, render_template, jsonify, redirect

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


app = Flask(__name__)

client = MongoClient()
# db_number = client['math_book_info']
# db_origin = client['math_exercise_origins']
# db_performance = client['math_performance']
db_vocab = client['vocab']

cols = ['_id', 'button', 'name', 'page', 'timestamp']

df_main = pd.DataFrame(list(db_vocab['Main'].find({'name': 'Calvin'})))
df_main['timestamp'] = df_main['timestamp'].apply(datetime.datetime.fromtimestamp)
print(df_main.head())

df_practice = pd.DataFrame(list(db_vocab['Practice'].find({'name': 'Calvin'})))
df_practice['timestamp'] = df_practice['timestamp'].apply(datetime.datetime.fromtimestamp)
print(df_practice.head())

df_quiz = pd.DataFrame(list(db_vocab['Quiz'].find({'name': 'Calvin'})))
df_quiz['timestamp'] = df_quiz['timestamp'].apply(datetime.datetime.fromtimestamp)
print(df_quiz.head())


df = df_main[cols].\
    append(df_practice[cols]).\
    append(df_quiz[cols]).\
    sort_values('timestamp')

print(df.head(200))

# todo: choose an arbitrary date to start this
# todo: determine a start and stop because sometimes the quit is hours after.
# todo:






# what should this data set look like?
#



if __name__ == '__main__':
    pass



