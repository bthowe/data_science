import os
import sys
import json
from pymongo import MongoClient
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

client = MongoClient()
db = client['math_book_info']
db_origins = client['math_exercise_origins']


@app.route("/")
def enter_chapter_details():
    return render_template('enter_exercises_origins.html')


@app.route('/query_chapter', methods=['POST', 'GET'])
def query_chapter():
    js = json.loads(request.data.decode('utf-8'))
    print(js)

    book = js['book']

    chapter_details = None
    for record in db[book].find({'chapter': js['chapter']}):
        chapter_details = {'num_lesson_probs': record['num_lesson_probs'], 'num_mixed_probs': record['num_mixed_probs']}
    print(chapter_details)

    return jsonify(chapter_details)


@app.route('/enter_origin_list', methods=['POST'])
def enter_origin_list():
    js = json.loads(request.data.decode('utf-8'))
    print(js)

    collection = db_origins[js['book']]
    y = collection.insert_one(js)
    print(y)

    print('data inserted: {}'.format(js))
    return ''


if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(host='0.0.0.0', port=8001, debug=True)
