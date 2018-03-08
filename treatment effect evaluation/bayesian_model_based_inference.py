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

    data['treatment'] = np.random.randint(0, 2, size=(len(data), 1))

    def treatment_maker(x):
        if x == 1:
            return 1
        else:
            if np.random.uniform(0, 1) < .25:
                return 1
            else:
                return 0
    data['income'] = data['income'].apply(treatment_maker)

    # print(income.value_counts())
    joblib.dump(data, 'data.pkl')

def model_parameter_distribution_estimate(data):
    """
    This function estimates the posterior probability distribution of the parameters of a GLM model with logit link
    function, where the outcome is distributed as a Bernoulli random variable.

    Prior: Uses the default prior specification for GLM---this is p(theta) = N(0, 10^{12} * I). This is a very vague
    prior that influences the posterior very little.
    Likelihood: The product of n Bernoulli trials, where the probability is given by the logistic function
    1 / (1 + e^{-z_i}), where z_i = X*beta.
    """
    data_t = data.query('treatment == 1')
    data_c = data.query('treatment == 0')

    with pm.Model() as logistic_model_t:
        pm.glm.GLM.from_formula('income ~ age + age2 + educ + hours', data_t, family=pm.glm.families.Binomial())  # drilling down into the Binomial class, there appears to be no other available link functions other than logit
        trace_logistic_model = pm.sample(2000, chains=1, tune=1000)
    joblib.dump(trace_logistic_model, 'trace_logistic_model_t.pkl')

    with pm.Model() as logistic_model_c:
        pm.glm.GLM.from_formula('income ~ age + age2 + educ + hours', data_c, family=pm.glm.families.Binomial())  # drilling down into the Binomial class, there appears to be no other available link functions other than logit
        trace_logistic_model = pm.sample(2000, chains=1, tune=1000)
    joblib.dump(trace_logistic_model, 'trace_logistic_model_c.pkl')

def posterior_predicted_distribution(trace_model, data):
    """
    Given the estimated distributions of the model parameters, this function estimates the posterior predicted
    distribution of the mean.

    :param trace_model: estimated distributions of the model parameters
    :param data: the data
    :return: posterior predicted distribution of the mean
    """
    trace = trace_model[1000:]

    lm = lambda x, samples: 1 / (1 + np.exp(-(samples['Intercept'] + samples['age']*x['age'] + samples['age2']*x['age2'] + samples['educ']*x['educ'] + samples['hours']*x['hours'])))  # todo: this isn't very general.

    # todo: do I need to loop or is there a slicker way?
    data['p'] = [np.random.choice(lm(row[1], trace)) for row in data.iterrows()]
    return data

def missing_draws(df, p):
    """
    Draws from the distribution of the missing data conditional on the observed data and the parameters.
    """
    df['income_missing'] = binom.rvs(1, df[p])
    return df

def outcomes_create(df):
    df['y_control'] = df.apply(lambda x: x['income'] if x['treatment'] == 0 else x['income_missing'], axis=1)
    df['y_treatment'] = df.apply(lambda x: x['income'] if x['treatment'] == 1 else x['income_missing'], axis=1)
    return df

def treatment_effect_calc(df):
    return (df['y_treatment'] - df['y_control']).mean(), (df['y_treatment'] - df['y_control']).std()


if __name__ == '__main__':
    # data_create()

    data = joblib.load('data.pkl')
    # model_parameter_distribution_estimate(data)

    trace_t = joblib.load('trace_logistic_model_t.pkl')
    trace_c = joblib.load('trace_logistic_model_c.pkl')
    treatment = posterior_predicted_distribution(trace_c, data.query('treatment == 1').iloc[:100]).pipe(missing_draws, 'p')
    control = posterior_predicted_distribution(trace_t, data.query('treatment == 0').iloc[:100]).pipe(missing_draws, 'p')

    tau = treatment.append(control).\
        pipe(outcomes_create).\
        pipe(treatment_effect_calc)

    print(tau)


# Objective: draw from the conditional distribution of Y^{mis}
# 1. Draw from the posterior distribution of the parameters (of this conditional distribution) given the observed data, Y^{obs} and W.
    # 1. estimate the posterior distribution of these parameters by estimating a model
        # 1. estimate the posterior distributions of this model parameters
        # 2. use these distributions to estimate the posterior predictid distribution, which are the parameters of the conditonal distribution
# 2. Draw from the conditional distribution of Y^{mis}
    # 1. For each observation, draw from the posterior distribution of the distribution parameters
    # 2. Draw from the conditional distribution, conditional on these parameter values.



# Quite obviously, if I'm estimating the posterior of the parameters (the means) using a logistic regression, then there is not a variance to estimate. In other words the conditional distribution of the missing data given the observed data and the parameters is bernoulli and not normal.
#
# todo: how should the treatment effect be measured if the outcome is binary?
# todo: I'm not sure I do things correctly here...am I estimating the treatment effect correctly?
