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

    start_chapter_details = list(db_number[book].find({'chapter': js['start_chapter']}))[0]
    print(start_chapter_details['num_lesson_probs'])
    print(start_chapter_details['num_mixed_probs'])
    end_chapter_details = list(db_number[book].find({'chapter': js['end_chapter']}))[0]
    print(end_chapter_details['num_lesson_probs'])
    print(end_chapter_details['num_mixed_probs'])

    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    problems_dic = {}
    # problems for beginning chapter
    if str(js['start_problem']).isalpha():
        problems_dic[js['start_chapter']] = [letter for letter in alphabet if letter >= str(js['start_problem']) and letter <= start_chapter_details['num_lesson_probs']] + \
                                            list(map(str, range(1, int(start_chapter_details['num_mixed_probs']) + 1)))
    else:
        problems_dic[js['start_chapter']] = [str(number) for number in range(1, int(start_chapter_details['num_mixed_probs']) + 1) if number >= int(js['start_problem'])]

    # problems for middling chapters
    if int(js['end_chapter']) - int(js['start_chapter']) > 1:
        for chapter in range(int(js['start_chapter']) + 1, int(js['end_chapter'])):
            mid_chapter_details = list(db_number[book].find({'chapter': chapter}))[0]
            problems_dic[chapter] = [letter for letter in alphabet if letter <= mid_chapter_details['num_lesson_probs']] + list(range(1, int(mid_chapter_details['num_mixed_probs']) + 1))

    # problems for end chapter
    if str(js['end_problem']).isalpha():
        problems_dic[js['end_chapter']] = [letter for letter in alphabet if letter <= str(js['end_problem'])]
    else:
        problems_dic[js['end_chapter']] = [letter for letter in alphabet if letter <= str(end_chapter_details['num_lesson_probs'])] +\
            [str(number) for number in range(1, int(end_chapter_details['num_mixed_probs']) + 1) if number <= int(js['end_problem'])]


    for chapter in range(js['start_chapter'], js['end_chapter'] + 1):
        problems_dic[chapter] = str(problems_dic[chapter])

    print(problems_dic)


    # return ''
    # return jsonify(items=[problems_dic])
    return jsonify(problems_dic)


# todo: I need to add date and start and end times to the post


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

# todo: style the pages
