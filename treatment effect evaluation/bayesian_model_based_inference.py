import sys
import joblib
import pymc3 as pm
import numpy as np
import pandas as pd
from scipy.stats import binom
import matplotlib.pyplot as plt



pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def data_create():
    data = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        header=None,
        names=['age', 'workclass', 'fnlwgt', 'education-categorical', 'educ', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'captial-gain', 'capital-loss', 'hours', 'native-country', 'income']
    )

    data = data[~pd.isnull(data['income'])]
    data[data['native-country']==" United-States"]

    income = 1 * (data['income'] == " >50K")
    age2 = np.square(data['age'])

    data = data[['age', 'educ', 'hours']]
    data['age2'] = age2
    data['income'] = income

    # print(income.value_counts())
    joblib.dump(data, 'data.pkl')

def model_train(data):
    with pm.Model() as logistic_model:
        pm.glm.GLM.from_formula('income ~ age + age2 + educ + hours', data, family=pm.glm.families.Binomial())  # drilling down into the Binomial class, there appears to be no other available link functions other than logit
        trace_logistic_model = pm.sample(2000, chains=1, tune=1000)

    joblib.dump(trace_logistic_model, 'trace_logistic_model.pkl')

def posterior_predicted(trace_model, data):
    trace = trace_model[1000:]

    lm = lambda x, samples: 1 / (1 + np.exp(-(samples['Intercept'] + samples['age']*x['age'] + samples['age2']*x['age2'] + samples['educ']*x['educ'] + samples['hours']*x['hours'])))

    # todo: do I need to loop or is there a slicker way?
    data['random_mu'] = [np.random.choice(lm(row[1], trace)) for row in data.iterrows()]
    return data

def missing_draws(df, p):
    df['y_missing'] = binom.rvs(1, df[p])
    return df


def treatment_effect_calc(df, treatment_var, control_var):
    return (df[treatment_var] - df[control_var]).mean(), (df[treatment_var] - df[control_var]).std()




if __name__ == '__main__':
    # data_create()

    data = joblib.load('data.pkl')
    # model_train(data)

    trace = joblib.load('trace_logistic_model.pkl')
    print(posterior_predicted(trace, data.iloc[:100]).pipe(missing_draws, 'random_mu').pipe(treatment_effect_calc, 'income', 'y_missing'))

# todo: which distribution do I choose in the function "missing_draws"?
# todo: how do I calculate the treatment effect when the outcome is binary?


# estimate a model separately for treatment and control
# draw randomly from the posterior predictive distribution of the mean for each, so each observation has a mean prediction.
# conditional on these drawn means, draw randomly for each of the missing values.
# calculate the treatment effect.
