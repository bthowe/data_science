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
    lesson_num = str(request.form['user_input'])
    # prompt_type = str(request.form['prompt_type'])

    d = {
        'user_image1': 'static/{0}/photo1.png'.format(lesson_num),
        'user_image2': 'static/{0}/photo2.png'.format(lesson_num)
    }
    return render_template("display_card.html", **d)  #, user_image1=full_filename1)

    # full_filename2 = 'static/{0}/photo2.png'.format(lesson_num)
    # return render_template(
    #     "display_card.html",
    #     user_image1='static/{0}/photo1.png'.format(lesson_num),
    #
    # )  #, user_image1=full_filename1)
    #

# @app.route('/')
# def show_index():
#     # lesson_num = str(request.form['user_input'])
#     # prompt_type = str(request.form['prompt_type'])
#
#     full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'photo2.png')
#     print(full_filename)
#     return render_template("display_card.html", user_image=full_filename)

    # return '''
    #     <img src="/photo2.png", alt="yo", height="200", width="200">
    #     '''


        # <img src="/Users/travis.howe/Desktop/rc/vocab_images_chapters/4/photo2.png">

        # <div class="nav3" style="height:705px;">
        #     <img src="photo2.png">
        #     <img src="photo2.png">
        # </div>
        # '''


# run the following:
# gunicorn --bind 0.0.0.0:8000 server:app

if __name__ == '__main__':
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