import os
import sys
import json
import joblib
import webbrowser
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.externals import joblib
from flask import Flask, request, render_template

app = Flask(__name__)
lesson_lst = list(range(4, 13)) + list(range(14, 75))

# client = MongoClient()
# # client.drop_database('vocab')
# # sys.exit()
# db = client['vocab']

# @app.before_first_request
# def browser_launch():
#     webbrowser.open('http://localhost:8001/')

import os
print(os.getcwd())

@app.route('/')
def submission_page():
    return render_template('login.html')

# @app.route('/practice', methods=['POST'])
# def practice():
#     lesson_num = str(request.form['user_input'])
#     prompt_type = str(request.form['prompt_type'])
#     num_cards = len(os.listdir('static/{0}'.format(lesson_num)))
#     if prompt_type == 'word':
#         cards = [['static/{0}/rc_vocab_{0}_{1}.png'.format(lesson_num, num),
#                   'static/{0}/rc_vocab_{0}_{1}.png'.format(lesson_num, num + 1)] for num in range(0, num_cards, 2)]
#     else:
#         cards = [['static/{0}/rc_vocab_{0}_{1}.png'.format(lesson_num, num + 1),
#                   'static/{0}/rc_vocab_{0}_{1}.png'.format(lesson_num, num)] for num in range(0, num_cards, 2)]
#
#     return render_template("display_card.html", cards=cards)
#
#
# @app.route('/quiz', methods=['POST'])
# def quiz():
#     practice_type = str(request.form['practice_type'])
#     lesson_num = str(request.form['user_input'])
#     prompt_type = str(request.form['prompt_type'])
#     num_cards = len(os.listdir('static/{0}'.format(lesson_num)))
#     if prompt_type == 'word':
#         cards = [['static/{0}/rc_vocab_{0}_{1}.png'.format(lesson_num, num),
#                   'static/{0}/rc_vocab_{0}_{1}.png'.format(lesson_num, num + 1)] for num in range(0, num_cards, 2)]
#
#         alternatives = []
#         for card_i in range(len(cards)):
#             alternatives_i = []
#             for j in range(3):
#                 random_lesson = np.random.choice(lesson_lst)
#                 num_cards_in_random_lesson = len(os.listdir('static/{0}'.format(random_lesson)))
#                 random_card = np.random.choice(range(0, num_cards_in_random_lesson, 2))
#                 alternatives_i.append('static/{0}/rc_vocab_{0}_{1}.png'.format(random_lesson, random_card + 1))
#             alternatives.append(alternatives_i)
#
#     else:
#         cards = [['static/{0}/rc_vocab_{0}_{1}.png'.format(lesson_num, num + 1),
#                   'static/{0}/rc_vocab_{0}_{1}.png'.format(lesson_num, num)] for num in range(0, num_cards, 2)]
#
#         alternatives = []
#         for card_i in range(len(cards)):
#             alternatives_i = []
#             for j in range(3):
#                 random_lesson = np.random.choice(lesson_lst)
#                 num_cards_in_random_lesson = len(os.listdir('static/{0}'.format(random_lesson)))
#                 random_card = np.random.choice(range(0, num_cards_in_random_lesson, 2))
#                 alternatives_i.append('static/{0}/rc_vocab_{0}_{1}.png'.format(random_lesson, random_card))
#             alternatives.append(alternatives_i)
#
#     if practice_type == 'practice':
#         return render_template('display_card.html', cards=cards)
#     else:
#         return render_template('quiz_card.html', cards=cards, alts=alternatives)
#
#
# @app.route('/mongo_call', methods=['POST'])
# def mongo_call():
#     js = json.loads(request.data.decode('utf-8'))
#
#     tab = db[js['page']]
#     tab.insert_one(js)
#
#     print('data inserted: {}'.format(js))
#     return ''
#
#
# def shutdown_server():
#     func = request.environ.get('werkzeug.server.shutdown')
#     if func is None:
#         raise RuntimeError('Not running with the Werkzeug Server')
#     func()
#
# @app.route('/quit')
# def quit():
#     shutdown_server()
#     return render_template('quit.html')
#

if __name__ == '__main__':
    cards = []
    discards = []
    quiz_count = 0

    app.run(host='0.0.0.0', port=8001, debug=True)
