import os
import sys
import json
import joblib
import datetime
import webbrowser
import numpy as np
import pandas as pd
from pymongo import MongoClient
from collections import defaultdict
from flask import Flask, request, render_template, jsonify, redirect

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


app = Flask(__name__)




@app.route("/")
def main_menu():
    return render_template('main_menu.html')


@app.route("/my_info", methods=['POST'])
def my_info():
    joblib.dump(dict(request.form), 'player_list.pkl')
    return render_template('my_info.html')


@app.route("/play", methods=['POST'])
def play():
    print(dict(request.form))
    print(joblib.load('player_list.pkl'))
    return render_template('play.html')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)
