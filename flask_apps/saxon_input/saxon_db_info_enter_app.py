import os
import sys
import json
from pymongo import MongoClient
from flask import Flask, request, render_template

app = Flask(__name__)

client = MongoClient()
# client.drop_database('math_book_info')
# sys.exit()
db = client['math_book_info']

@app.route("/")
def enter_chapter_details():
    return render_template('enter_chapter_details.html')

@app.route('/add_document', methods=['POST'])
def add_document():
    js = json.loads(request.data.decode('utf-8'))

    collection = db[js['book'].replace(' ', '_').replace('/', '_')]
    y = collection.insert_one(js)
    print(y)

    print('data inserted: {}'.format(js))
    return ''

@app.route('/remove_document', methods=['POST'])
def remove_document():
    js = json.loads(request.data.decode('utf-8'))

    collection = db[js['book'].replace(' ', '_').replace('/', '_')]
    y = collection.delete_one(js)
    print(y)

    print('data deleted: {}'.format(js))
    return ''

if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(host='0.0.0.0', port=8001, debug=True)
