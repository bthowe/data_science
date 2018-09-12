import os
import sys
import json
import joblib
import webbrowser
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.externals import joblib
from flask import Flask, request, render_template, session, flash

app = Flask(__name__)

client = MongoClient()
# client.drop_database('math_book_info')
# sys.exit()
db = client['math_book_info']

@app.route("/")
def enter_chapter_details():
    book = 'Math 5/4'
    return render_template('enter_chapter_details.html', book=book)


@app.route('/mongo_call', methods=['POST'])
def mongo_call():
    js = json.loads(request.data.decode('utf-8'))

    collection = db[js['book'].replace(' ', '_').replace('/', '_')]
    y = collection.insert_one(js)
    print(y)

    # cursor = collection.find()
    # for record in cursor:
    #     print(record)

    print('data inserted: {}'.format(js))
    return ''

if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(host='0.0.0.0', port=8001, debug=True)
