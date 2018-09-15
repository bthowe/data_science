import os
import sys
import json
from pymongo import MongoClient
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

client = MongoClient()
db = client['math_book_info']
db_record = client['math_performance']

@app.route("/")
def enter_chapter_details():
    return render_template('enter_exercises_details.html')

@app.route('/query_chapter', methods=['POST', 'GET'])
def add_document():
    js = json.loads(request.data.decode('utf-8'))
    print(js)

    book = js['book']

    for record in db[book].find({'chapter': js['chapter']}):
        chapter_details = {'num_lesson_probs': record['num_lesson_probs'], 'num_mixed_probs': record['num_mixed_probs']}
    print(chapter_details)

    return jsonify(chapter_details)


@app.route('/remove_document', methods=['POST'])
def remove_document():
    js = json.loads(request.data.decode('utf-8'))

    collection = db[js['book'].replace(' ', '_').replace('/', '_')]
    y = collection.delete_one(js)
    print(y)

    print('data deleted: {}'.format(js))
    return ''


@app.route('/missed_problems', methods=['POST'])
def missed_problems():
    js = json.loads(request.data.decode('utf-8'))
    print(js)

    collection = db_record[js['book']]
    y = collection.insert_one(js)
    print(y)

    print('data inserted: {}'.format(js))
    return ''


if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(host='0.0.0.0', port=8001, debug=True)
