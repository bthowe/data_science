import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
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
    return data


def fi(X, y, n):
    """
    Calculates the average increase in R squared for each feature. This function does the following: (1) An ordering of
    the features is chosen; (2) features are successively added in order to the set of features used to fit a model; (3)
    and the increase in R squared is found after each iteration; (4) the average increase in R squared is calculated
    for each variable.

    :param X: Covariates...pandas dataframe
    :param y: Outcomes...pandas series
    :param n: number of times each feature is scored
    :return: Dictionary of features and their average increase on the outcome in terms of the R squared
    """
    cols = X.columns.tolist()

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

    X = df
    y = df.pop('income')
    n = 10
    fi(X, y, n)
