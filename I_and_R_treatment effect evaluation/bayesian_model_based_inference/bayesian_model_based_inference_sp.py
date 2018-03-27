import sys
import joblib
import pymc3 as pm
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def data_create():
    b0 = 4.3
    b1 = -6.7
    b2 = 7.9
    b3 = -10.2
    b4 = 15.3

    df = pd.DataFrame(np.random.uniform(-1, 1, size=(1000, 4)), columns=['one', 'two', 'three', 'four'])
    df['treatment'] = np.random.randint(0, 2, size=(len(df), 1))

    def treatment_effect(x):
        if x['treatment'] == 0:
            return b0 + x['one'] * b1 + x['two'] * b2 + x['three'] * b3 + x['four'] * b4
        else:
            return 5 + b0 + x['one'] * b1 + x['two'] * b2 + x['three'] * b3 + x['four'] * b4
    df['mu'] = df.apply(treatment_effect, axis=1)
    df['sigma'] = 2
    df['target'] = norm.rvs(df['mu'], df['sigma'])
    df.drop(['mu', 'sigma'], 1, inplace=True)

    joblib.dump(df, 'data_normal.pkl')

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


    with pm.Model() as model_t:
        pm.glm.GLM.from_formula('target ~ one + two + three + four', data_t)
        trace = pm.sample(3000, cores=2)  # draw 3000 posterior samples using NUTS sampling
    joblib.dump(trace, 'trace_normal_model_t.pkl')

    with pm.Model() as model_t:
        pm.glm.GLM.from_formula('target ~ one + two + three + four', data_c)
        trace = pm.sample(3000, cores=2)  # draw 3000 posterior samples using NUTS sampling
    joblib.dump(trace, 'trace_normal_model_c.pkl')

def posterior_predicted_distribution(trace_model, data):
    """
    Given the estimated distributions of the model parameters, this function estimates the posterior predicted
    distribution of the mean.

    :param trace_model: estimated distributions of the model parameters
    :param data: the data
    :return: posterior predicted distribution of the mean
    """
    trace = trace_model[1000:]

    lm = lambda x, samples: samples['Intercept'] + samples['one']*x['one'] + samples['two']*x['two'] + samples['three']*x['three'] + samples['four']*x['four']  # todo: this isn't very general.

    # todo: do I need to loop or is there a slicker way?
    data['mu'] = [np.random.choice(lm(row[1], trace)) for row in data.iterrows()]
    data['sigma'] = np.random.choice(trace['sd'], size=(len(data), 1), replace=True)

    return data

def missing_draws(df):
    """
    Draws from the distribution of the missing data conditional on the observed data and the parameters.
    """
    df['target_missing'] = df['mu']  # only difference between super-population and finite-sample!
    return df

def outcomes_create(df):
    df['y_control'] = df.apply(lambda x: x['target'] if x['treatment'] == 0 else x['target_missing'], axis=1)
    df['y_treatment'] = df.apply(lambda x: x['target'] if x['treatment'] == 1 else x['target_missing'], axis=1)
    return df

def treatment_effect_calc(df):
    return (df['y_treatment'] - df['y_control']).mean(), (df['y_treatment'] - df['y_control']).std()

if __name__ == '__main__':
    # data_create()

    data = joblib.load('data_normal.pkl')
    # model_parameter_distribution_estimate(data)

    trace_t = joblib.load('trace_normal_model_t.pkl')
    trace_c = joblib.load('trace_normal_model_c.pkl')
    treatment = posterior_predicted_distribution(trace_c, data.query('treatment == 1')).pipe(missing_draws)
    control = posterior_predicted_distribution(trace_t, data.query('treatment == 0')).pipe(missing_draws)

    tau = treatment.append(control).\
        pipe(outcomes_create).\
        pipe(treatment_effect_calc)

    print(tau)
