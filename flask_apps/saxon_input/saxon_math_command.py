import os
import sys
import json
from pymongo import MongoClient
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

client = MongoClient()
db_number = client['math_book_info']
db_origin = client['math_exercise_origins']
db_performance = client['math_performance']

@app.route("/")
def main_menu():
    return render_template('main_menu.html')

@app.route("/enter_problem_number")
def enter_chapter_details():
    return render_template('enter_problem_number.html')

@app.route('/add_problem_number', methods=['POST'])
def add_problem_number():
    js = json.loads(request.data.decode('utf-8'))

    collection = db_number[js['book'].replace(' ', '_').replace('/', '_')]
    y = collection.insert_one(js)
    print(y)

    print('number data inserted: {}'.format(js))
    return ''

@app.route('/remove_problem_number', methods=['POST'])
def remove_problem_number():
    js = json.loads(request.data.decode('utf-8'))

    collection = db_number[js['book'].replace(' ', '_').replace('/', '_')]
    y = collection.delete_one(js)
    print(y)

    print('number data deleted: {}'.format(js))
    return ''

@app.route("/enter_problem_origin")
def enter_problem_origin():
    return render_template('enter_problem_origins.html')


@app.route('/query_chapter', methods=['POST', 'GET'])
def query_chapter():
    js = json.loads(request.data.decode('utf-8'))
    print(js)

    book = js['book']

    chapter_details = None
    for record in db_number[book].find({'chapter': js['chapter']}):
        chapter_details = {'num_lesson_probs': record['num_lesson_probs'], 'num_mixed_probs': record['num_mixed_probs']}
    print(chapter_details)

    return jsonify(chapter_details)


@app.route('/add_problem_origin', methods=['POST'])
def add_problem_origin():
    js = json.loads(request.data.decode('utf-8'))
    print(js)

    collection = db_origin[js['book']]
    y = collection.insert_one(js)
    print(y)

    print('data inserted: {}'.format(js))
    return ''


@app.route("/enter_performance")
def enter_performance():
    return render_template('enter_performance.html')


@app.route('/add_missed_problems', methods=['POST'])
def add_missed_problems():
    js = json.loads(request.data.decode('utf-8'))
    print(js)

    collection = db_performance[js['book']]
    y = collection.insert_one(js)
    print(y)

    print('data inserted: {}'.format(js))
    return ''


if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(host='0.0.0.0', port=8001, debug=True)

# todo: put a main menu button in each of the three pages
# todo: css the main page
