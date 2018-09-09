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
lesson_lst = list(range(4, 13)) + list(range(14, 75))

# client = MongoClient()
# # client.drop_database('vocab')
# # sys.exit()
# db = client['vocab']

# @app.before_first_request
# def browser_launch():
#     webbrowser.open('http://localhost:8001/')


@app.route('/')
def home():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        return "Hello Boss!  <a href='/logout'>Logout</a>"


@app.route('/login', methods=['POST'])
def do_admin_login():
    if request.form['password'] == 'password' and request.form['username'] == 'admin':
        session['logged_in'] = True
    else:
        flash('wrong password!')
    return home()


@app.route("/register")
def register():
    return render_template('register.html')


@app.route("/logout")
def logout():
    session['logged_in'] = False
    return home()

if __name__ == '__main__':
    cards = []
    discards = []
    quiz_count = 0

    app.secret_key = os.urandom(12)
    app.run(host='0.0.0.0', port=8001, debug=True)
