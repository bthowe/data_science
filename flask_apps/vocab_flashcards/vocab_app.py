import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)


@app.route('/')
def submission_page():
    return '''
        <form action="/quiz" method='POST' >
            <LABEL for="prompt_type">Prompt Type: </LABEL> <br>
            <input type="radio" name="prompt_type" value="word" checked> Word <br>
            <input type="radio" name="prompt_type" value="def"> Definition/Sentence <br> <br>

            <LABEL for="chapter">Lesson Number: </LABEL>
            <input type="text" name="user_input" />
            <input type="submit" />
        </form>
        '''


@app.route('/quiz', methods=['POST'])
def quiz():
    global quiz_count, cards, discards
    quiz_count += 1


    # print(str(request.form['next']))

    if quiz_count == 1:
        lesson_num = str(request.form['user_input'])
        prompt_type = str(request.form['prompt_type'])
        num_cards = len(os.listdir('static/{0}'.format(lesson_num)))
        if prompt_type == 'word':
            cards = [('static/{0}/rc_vocab_{0}_{1}.png'.format(lesson_num, num),
                      'static/{0}/rc_vocab_{0}_{1}.png'.format(lesson_num, num + 1)) for num in range(0, num_cards, 2)]
        else:
            cards = [('static/{0}/rc_vocab_{0}_{1}.png'.format(lesson_num, num + 1),
                      'static/{0}/rc_vocab_{0}_{1}.png'.format(lesson_num, num)) for num in range(0, num_cards, 2)]
        discards = []

    if cards:
        card = cards.pop(0)
        discards.append(card)

        d = {
            'user_image1': card[0],
            'user_image2': card[1],
        }
        return render_template("display_card.html", **d)
    else:
        quiz_count = 0
        return render_template("finished.html")


if __name__ == '__main__':
    cards = []
    discards = []
    quiz_count = 0

    app.run(host='0.0.0.0', port=8000, debug=True)


# intro page:
# buttons:
# - choose sentence or definition
# - ask definition to word or word to definition (can I then udpate this from the choice of sentence or definition)
# - choose book
# - groups of five, ten, etc.

# vocab page:
# buttons:
# back to intro page
# flip card
# next
# previous
# repeat group of five, etc.

# display:
# two panels: word or definition on the left and the other on the right.

# todo: css, split the definition/sentence card