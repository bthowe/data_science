import sys
import joblib
import pymc3 as pm
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import binom
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

def data_create():
    data = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        header=None,
        names=['age', 'workclass', 'fnlwgt', 'education-categorical', 'educ', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'captial-gain', 'capital-loss', 'hours', 'native-country', 'income']
    )

    data = data[~pd.isnull(data['income'])].sample(n=100)
    data[data['native-country']==" United-States"]

    age2 = np.square(data['age'])

    data = data[['age', 'educ', 'hours']]
    data['age2'] = age2

    beta0 = 3.5
    beta1 = 34.7
    beta2 = 65.2
    beta3 = 2398.23
    beta4 = 891.263
    eps = norm.rvs(0, 1000, size=(len(data), ))
    data['income'] = beta0 + data['age'] * beta1 + data['age2'] * beta2 + data['educ'] * beta3 + data['hours'] * beta4 + eps
    # joblib.dump(data, 'data.pkl')
    return data


def fi(X, y):
    cols = X.columns.tolist()
    n = 10

    feature_dict = {col: 0 for col in cols}

    for iter in range(n):
        baseline_metric_score = 0
        model_features = []
        for feature in np.random.choice(cols, len(cols), replace=False):
            model_features.append(feature)

            lr = LinearRegression()
            lr.fit(X[model_features], y)

            baseline_metric_score_new = r2_score(y, lr.predict(X[model_features]))
            feature_dict[feature] += baseline_metric_score_new - baseline_metric_score
            baseline_metric_score = baseline_metric_score_new

    feature_dict = {k: v / n for k, v in feature_dict.items()}
    print(feature_dict)

if __name__ == '__main__':
    df = data_create()
    # data = joblib.load('data.pkl')

    X = df
    y = df.pop('income')
    fi(X, y)
