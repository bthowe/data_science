"""Filename: server.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def apicall():
    """API Call

    Pandas dataframe (sent as a payload) from API Call
    """
    try:
        test_json = request.get_json()
        test = pd.read_json(test_json, orient='records')

    except Exception as e:
        raise e

    if test.empty:
        return "bad"
    else:
        print("Loading the model...")
        model = joblib.load('models/model.pkl')

        print("The model has been loaded...doing predictions now...")
        predictions = pd.DataFrame(model.predict_proba(test)[:, 1], columns=['predictions'])

        responses = jsonify(predictions=predictions.to_json(orient="records"))
        responses.status_code = 200

        return responses


# run the following:
# gunicorn --bind 0.0.0.0:8000 server:app

if __name__ == '__main__':
    np.random.seed(4)
    df = pd.concat(
        [
            pd.DataFrame(np.random.uniform(0, 1, size=(200, 3)), columns=['one', 'two', 'three']),
            pd.DataFrame(np.random.randint(0, 2, size=(200, 1)), columns=['target'])
        ], axis=1
    )

    X = df
    y = X.pop('target')

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(X, y)
    joblib.dump(lr, 'models/model.pkl')


# todo: if test.empty: return "bad"...this needs to be fixed somehow

