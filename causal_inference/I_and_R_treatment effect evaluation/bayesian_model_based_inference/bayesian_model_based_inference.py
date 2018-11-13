import sys
import joblib
import pymc3 as pm
import numpy as np
import pandas as pd
from scipy.stats import binom, norm
import matplotlib.pyplot as plt

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def data_create():
    N = 1000

    b0 = -1.6
    b1 = -0.03
    b2 = 0.6
    b3 = 1.6

    df = pd.DataFrame(np.random.uniform(-1, 1, size=(N, 3)), columns=['one', 'two', 'three'])
    df['treatment'] = np.random.randint(0, 2, size=(len(df), 1))

    def sigmoid(x):
        return np.exp(x) / (1 + np.exp(x))

    def treatment_effect(x):
        if x['treatment'] == 0:
            return sigmoid(b0 + x['one'] * b1 + x['two'] * b2 + x['three'] * b3)
        else:
            return sigmoid(3 + b0 + x['one'] * b1 + x['two'] * b2 + x['three'] * b3)
    df['p'] = df.apply(treatment_effect, axis=1)
    df['y'] = binom.rvs(1, df['p'])

    print('Estimate of treatment effect: {}'.format(df.query('treatment == 1')['p'].mean() - df.query('treatment == 0')['p'].mean()))

    nt = df.query('treatment == 1').shape[0]
    nc = df.query('treatment == 0').shape[0]
    p = df['y'].mean()

    print('Estimate of standard error: {}'.format(np.sqrt(p * (1-p) * ((1 / nt) + (1 / nc)))))
    print('\n\n')
    print('Difference in means: {}'.format(df.query('treatment == 1')['y'].mean() - df.query('treatment == 0')['y'].mean()))

    df['constant'] = 1

    joblib.dump(df, 'data.pkl')
    sys.exit()


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
        pm.glm.GLM.from_formula('y ~ one + two + three', data_t, family=pm.glm.families.Binomial())  # drilling down into the Binomial class, there appears to be no other available link functions other than logit
        trace_logistic_model = pm.sample(2000, chains=1, tune=1000)
    joblib.dump(trace_logistic_model, 'trace_logistic_model_t.pkl')

    with pm.Model() as logistic_model_c:
        pm.glm.GLM.from_formula('y ~ one + two + three', data_c, family=pm.glm.families.Binomial())  # drilling down into the Binomial class, there appears to be no other available link functions other than logit
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

    lm = lambda x, samples: 1 / (1 + np.exp(-(samples['Intercept'] + samples['one']*x['one'] + samples['two']*x['two'] + samples['three']*x['three'])))  # todo: this isn't very general.

    # todo: do I need to loop or is there a slicker way?
    data['p'] = [np.random.choice(lm(row[1], trace)) for row in data.iterrows()]
    return data

def missing_draws(df, p):
    """
    Draws from the distribution of the missing data conditional on the observed data and the parameters.
    """
    df['y_missing'] = binom.rvs(1, df[p])
    return df

def outcomes_create(df):
    df['y_control'] = df.apply(lambda x: x['y'] if x['treatment'] == 0 else x['y_missing'], axis=1)
    df['y_treatment'] = df.apply(lambda x: x['y'] if x['treatment'] == 1 else x['y_missing'], axis=1)
    return df

def proportions_sd(df):
    p1 = df['y_treatment'].mean()
    n1 = n2 = len(df)
    p2 = df['y_control'].mean()
    return np.sqrt(((p1 * (1 - p1)) / n1) + ((p2 * (1 - p2)) / n2))

def treatment_effect_calc(df):
    return (df['y_treatment'] - df['y_control']).mean(), proportions_sd(df)
    # return (df['y_treatment'] - df['y_control']).mean(), (df['y_treatment'] - df['y_control']).std()

if __name__ == '__main__':
    # data_create()

    data = joblib.load('data.pkl')

    model_parameter_distribution_estimate(data)

    trace_t = joblib.load('trace_logistic_model_t.pkl')
    trace_c = joblib.load('trace_logistic_model_c.pkl')
    treatment = posterior_predicted_distribution(trace_c, data.query('treatment == 1').iloc[:100]).pipe(missing_draws, 'p')
    control = posterior_predicted_distribution(trace_t, data.query('treatment == 0').iloc[:100]).pipe(missing_draws, 'p')

    tau = treatment.append(control).\
        pipe(outcomes_create).\
        pipe(treatment_effect_calc)

    print(tau)
