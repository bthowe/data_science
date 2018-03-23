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
    beta = 2
    gamma = 1
    mu = 1
    sigma_mu = .1
    sigma_c = .2
    sigma_t = .3

    np.random.seed(seed=2)
    N = 100
    df_t = pd.DataFrame(np.random.uniform(-1, 1, size=(N, 1)), columns=['one'])
    df_t['stratum'] = np.arange(N)
    df_t['treatment'] = 1
    df_c = pd.DataFrame(np.random.uniform(-1, 1, size=(N, 1)), columns=['one'])
    df_c['stratum'] = np.arange(N)
    df_c['treatment'] = 0

    df = df_t.append(df_c)
    df['mu_j'] = df.groupby(df['stratum']).apply(lambda x: np.random.normal(mu, sigma_mu))

    def y(x):
        if x['treatment'] == 1:
            return np.random.normal(beta * x['one'] + gamma + x['mu_j'], sigma_t)
        else:
            return np.random.normal(beta * x['one'] + x['mu_j'], sigma_c)
    df['y'] = df.apply(y, axis=1)
    return df

def model_parameter_distribution_estimate(data):
    """
    This function estimates the posterior probability distribution of the parameters of a hierarchical model.
    """
    strata = data['stratum'].values
    n_strata = len(data['stratum'].unique())

    treat = data['treatment'].values
    n_treat = len(data['treatment'].unique())

    with pm.Model() as model_t:
        mu = pm.Normal('mu', mu=0., sd=100 ** 2)
        sigma_mu = pm.HalfCauchy('sigma_mu', 5)
        mu_j = pm.Normal('mu_j', mu=mu, sd=sigma_mu, shape=n_strata)

        sigma = pm.HalfCauchy('sigma', 5, shape=n_treat)

        beta = pm.Normal('beta', mu=0., sd=100 ** 2)
        gamma = pm.Normal('gamma', mu=0., sd=100 ** 2)

        y_est = mu_j[strata] + gamma * data['treatment'] + beta * data['one']
        sigma_est = sigma[treat]

        y_like = pm.Normal('y_like', mu=y_est, sd=sigma_est, observed=data['y'])

        trace = pm.sample(target_accept=.95)
        # trace = pm.sample(draws=10000, n_init=5000)

        # pm.traceplot(trace)
        # plt.show()
        # print(trace.varnames)

        joblib.dump(trace, '/Users/travis.howe/Downloads/trace.pkl')


def posterior_predicted_distribution(trace, data):
    """
    Given the estimated distributions of the model parameters, this function estimates the posterior predicted
    distribution of the mean.

    :param trace_model: estimated distributions of the model parameters
    :param data: the data
    :return: posterior predicted distribution of the mean
    """
    # todo: how do I get the sub mu_js out of the trace?
    #   -this must be done by index, determined by order of appearance. 


    # trace = trace_model[1000:]


    c_data = data.ix[data.county == c]
    c_data = c_data.reset_index(drop=True)
    c_index = np.where(county_names == c)[0][0]
    z = list(c_data['county_code'])[0]


    print(trace.varnames)
    print(trace['mu_j']['0'])  #[1000:][z]
    sys.exit()

    lm = lambda x, samples: samples['mu_j'] + samples['gamma'] * x['treatment'] + samples['beta'] * x['one']

    data['mu'] = [np.random.choice(lm(row[1], trace)) for row in data.iterrows()]
    data['sigma'] = np.random.choice(trace['sd'], size=(len(data), 1), replace=True)

    return data

def missing_draws(df):
    """
    Draws from the distribution of the missing data conditional on the observed data and the parameters.
    """
    df['target_missing'] = norm.rvs(df['mu'], df['sigma'])
    return df

def outcomes_create(df):
    df['y_control'] = df.apply(lambda x: x['target'] if x['treatment'] == 0 else x['target_missing'], axis=1)
    df['y_treatment'] = df.apply(lambda x: x['target'] if x['treatment'] == 1 else x['target_missing'], axis=1)
    return df

def treatment_effect_calc(df):
    return (df['y_treatment'] - df['y_control']).mean(), (df['y_treatment'] - df['y_control']).std()



if __name__ == '__main__':
    df = data_create()
    # model_parameter_distribution_estimate(df)

    trace = joblib.load('/Users/travis.howe/Downloads/trace.pkl')
    posterior_predicted_distribution(trace, df)


# todo: https://en.wikipedia.org/wiki/Inverse-gamma_distribution; inverse chi 2 can be cast as an inverse gamma
# todo: how do you go from the traces to the treatment effect? In finite sample, the same routine as above applies, whereas in super population, the posterior of gamma is what I want.
