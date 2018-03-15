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

    strata = data_t['stratum'].values
    n_strata = len(data_t['stratum'].unique())

    # county_names = data.county.unique()
    # county_idx = data.county_code.values
    #
    # n_counties = len(data.county.unique())
    #
    # with pm.Model() as model_t:
    #     mu_b0 = pm.Normal('mu_b0', mu=0., sd=100 ** 2)
    #     sigma_b0 = pm.HalfCauchy('sigma_b0', 5)
    #     mu_b1 = pm.Normal('mu_b1', mu=0., sd=100 ** 2)
    #     sigma_b1 = pm.HalfCauchy('sigma_b1', 5)
    #
    #     b0 = pm.Normal('b0', mu=mu_b0, sd=sigma_b0, shape=n_counties)
    #     b1 = pm.Normal('b1', mu=mu_b1, sd=sigma_b1, shape=n_counties)
    #
    #     eps = pm.HalfCauchy('eps', 5)
    #
    #     radon_est = b0[county_idx] + b1[county_idx] * data.floor.values
    #
    #     # Data likelihood
    #     radon_like = pm.Normal('radon_like', mu=radon_est, sd=eps, observed=data.log_radon)
    #
    #     trace = pm.sample(draws=2000, n_init=1000, cores=1, chains=1)
    #
    #     print(trace)
    #     # sys.exit()
    #     pm.traceplot(trace)
    #     plt.show()
    # sys.exit()

    features = ['intercept', 'one', 'two', 'three', 'treatment']
    with pm.Model() as model_t:
        # mu_b0 = pm.Normal('mu_b0', mu=0., sd=100 ** 2)
        # sigma_b0 = pm.HalfCauchy('sigma_b0', 5)
        # mu_b1 = pm.Normal('mu_b1', mu=0., sd=100 ** 2)
        # sigma_b1 = pm.HalfCauchy('sigma_b1', 5)
        # mu_b2 = pm.Normal('mu_b2', mu=0., sd=100 ** 2)
        # sigma_b2 = pm.HalfCauchy('sigma_b2', 5)
        # mu_b3 = pm.Normal('mu_b3', mu=0., sd=100 ** 2)
        # sigma_b3 = pm.HalfCauchy('sigma_b3', 5)
        # mu_btreatment = pm.Normal('mu_btreatment', mu=0., sd=100 ** 2)
        # sigma_btreatment= pm.HalfCauchy('sigma_btreatment', 5)
        #
        # b0 = pm.Normal('b0', mu=mu_b0, sd=sigma_b0, shape=n_strata)
        # b1 = pm.Normal('b1', mu=mu_b1, sd=sigma_b1, shape=n_strata)
        # b2 = pm.Normal('b2', mu=mu_b2, sd=sigma_b2, shape=n_strata)
        # b3 = pm.Normal('b3', mu=mu_b3, sd=sigma_b3, shape=n_strata)
        # btreatment = pm.Normal('btreatment', mu=mu_btreatment, sd=sigma_btreatment, shape=n_strata)

        # mu = {
        #     'mu_intercept': mu_b0,
        #     'mu_one': mu_b1,
        #     'mu_two': mu_b2,
        #     'mu_three': mu_b3,
        #     'mu_treatment': mu_btreatment
        # }
        # sigma = {
        #     'sigma_intercept': sigma_b0,
        #     'sigma_one': sigma_b1,
        #     'sigma_two': sigma_b2,
        #     'sigma_three': sigma_b3,
        #     'sigma_treatment': sigma_btreatment
        # }

        # b = {
        #     'b_intercept': pm.Normal('b0', mu=mu['mu_intercept'], sd=sigma['sigma_intercept'], shape=n_strata),
        #     'b_one': pm.Normal('b1', mu=mu['mu_one'], sd=sigma['sigma_one'], shape=n_strata),
        #     'b_two': pm.Normal('b2', mu=mu['mu_two'], sd=sigma['sigma_two'], shape=n_strata),
        #     'b_three': pm.Normal('b3', mu=mu['mu_three'], sd=sigma['sigma_three'], shape=n_strata),
        #     'b_treatment': pm.Normal('btreatment', mu=mu['mu_treatment'], sd=sigma['sigma_treatment'], shape=n_strata)
        # }
        # b = {
        #     'b_intercept': pm.Normal('b0', mu=mu_b0, sd=sigma_b0, shape=n_strata),
        #     'b_one': pm.Normal('b1', mu=mu_b1, sd=sigma_b1, shape=n_strata),
        #     'b_two': pm.Normal('b2', mu=mu_b2, sd=sigma_b2, shape=n_strata),
        #     'b_three': pm.Normal('b3', mu=mu_b3, sd=sigma_b3, shape=n_strata),
        #     'b_treatment': pm.Normal('btreatment', mu=mu_btreatment, sd=sigma_btreatment, shape=n_strata)
        # }
        # b = {
        #     'b0': b0,
        #     'b1': b1,
        #     'b2': b2,
        #     'b3': b3,
        #     'btreatment': btreatment
        # }

        # sys.exit()

        # mu = {'mu_{}'.format(col): pm.Normal('mu_{}'.format(col), mu=0., sd=100 ** 2) for col in features}
        # sigma = {'sigma_{}'.format(col): pm.Normal('sigma_{}'.format(col), mu=0., sd=100 ** 2) for col in features}
        # b = {'b_{}'.format(col): pm.Normal('b_{}'.format(col), mu=mu['mu_{}'.format(col)], sd=sigma['sigma_{}'.format(col)], shape=n_strata) for col in features}
        #
        # mu_b0 = pm.Normal('mu_b0', mu=0., sd=100 ** 2)
        # sigma_b0 = pm.HalfCauchy('sigma_b0', 5)
        # mu_b1 = pm.Normal('mu_b1', mu=0., sd=100 ** 2)
        # sigma_b1 = pm.HalfCauchy('sigma_b1', 5)
        # mu_b2 = pm.Normal('mu_b2', mu=0., sd=100 ** 2)
        # sigma_b2 = pm.HalfCauchy('sigma_b2', 5)
        # mu_b3 = pm.Normal('mu_b3', mu=0., sd=100 ** 2)
        # sigma_b3 = pm.HalfCauchy('sigma_b3', 5)
        # mu_btreatment = pm.Normal('mu_btreatment', mu=0., sd=100 ** 2)
        # sigma_btreatment= pm.HalfCauchy('sigma_btreatment', 5)

        # b0 = pm.Normal('b0', mu=mu_b0, sd=sigma_b0, shape=n_strata)
        # b1 = pm.Normal('b1', mu=mu_b1, sd=sigma_b1, shape=n_strata)
        # b2 = pm.Normal('b2', mu=mu_b2, sd=sigma_b2, shape=n_strata)
        # b3 = pm.Normal('b3', mu=mu_b3, sd=sigma_b3, shape=n_strata)
        # btreatment = pm.Normal('btreatment', mu=mu_btreatment, sd=sigma_btreatment, shape=n_strata)

        # y_est = b0[strata] + b1[strata] * data_t['one'] + b2[strata] * data_t['two'] + b3[strata] * data_t['three'] + btreatment[strata] * data_t['treatment']
        # y_est = b['b0'][strata] + b['b1'][strata] * data_t['one'] + b['b2'][strata] * data_t['two'] + b['b3'][strata] * data_t['three'] + b['btreatment'][strata] * data_t['treatment']

        mu = {'mu_{}'.format(col): pm.Normal('mu_{}'.format(col), mu=0., sd=100 ** 2) for col in features}
        sigma = {'sigma_{}'.format(col): pm.HalfCauchy('sigma_{}'.format(col), 5) for col in features}
        b = {'b_{}'.format(col): pm.Normal('b_{}'.format(col), mu=mu['mu_{}'.format(col)], sd=sigma['sigma_{}'.format(col)], shape=n_strata) for col in features}

        eps = pm.HalfCauchy('eps', 5)

        y_est = b['b_intercept'][strata]
        for col in [feature for feature in features if feature != 'intercept']:
            print(col)
            y_est += b['b_{}'.format(col)][strata] * data_t[col]

        y_like = pm.Normal('y_like', mu=y_est, sd=eps, observed=data_t['y'])

        trace = pm.sample(draws=2000, n_init=1000, chains=1)

        pm.traceplot(trace)
        plt.show()
        # plt.savefig('h1.png')


    sys.exit()
    # with pm.Model() as model_t:
#         mu_b0 = pm.Normal('mu_b0', mu=0., sd=100 ** 2)
#         sigma_b0 = pm.HalfCauchy('sigma_b0', 5)
#         mu_b1 = pm.Normal('mu_b1', mu=0., sd=100 ** 2)
#         sigma_b1 = pm.HalfCauchy('sigma_b1', 5)
    #
    #     b0 = pm.Normal('b0', mu=mu_b0, sd=sigma_b0, shape=n_strata)
    #     b1 = pm.Normal('b1', mu=mu_b1, sd=sigma_b1, shape=n_strata)
    #
    #     eps = pm.HalfCauchy('eps', 5)
    #
    #     radon_est = b0[strata] + b1[strata] * data_t['one']
    #
    #     # Data likelihood
    #     radon_like = pm.Normal('radon_like', mu=radon_est, sd=eps, observed=data_t['y'])
    #
    #     trace = pm.sample(draws=2000, n_init=1000, chains=1)
    #
    #     pm.traceplot(trace)
    #     plt.show()
    #     # plt.savefig('h1.png')
    #
    #
    # sys.exit()

    with pm.Model() as model_t:
        # todo: how do I specify the prior for the model standard deviation?

        mu_b0 = pm.Normal('mu_b0', mu=0, sd=100 ** 2)
        sigma_b0 = pm.HalfCauchy('sigma_b0', 5)
        mu_b1 = pm.Normal('mu_b1', mu=0, sd=100 ** 2)
        sigma_b1 = pm.HalfCauchy('sigma_b1', 5)

        b0 = pm.Normal.dist(mu=mu_b0, sd=sigma_b0)
        b1 = pm.Normal.dist(mu=mu_b1, sd=sigma_b1)

        priors = {'Intercept': b0, 'Regressor': b1}
        # pm.glm.GLM.from_formula('y ~ one', data_t, priors=priors)
        pm.glm.GLM.from_formula('y ~ one + two + three', data_t, priors=priors)

        trace = pm.sample(draws=2000, n_init=1000, chains=1)  # draw 3000 posterior samples using NUTS sampling

        pm.traceplot(trace)
        plt.show()
        # plt.savefig('h2.png')

    # with pm.Model() as model_t:
        # pm.glm.GLM.from_formula('target ~ one + two + three + four', data_t)
    #     trace = pm.sample(3000, cores=2)  # draw 3000 posterior samples using NUTS sampling
    # joblib.dump(trace, 'trace_normal_model_t.pkl')
    #
    # with pm.Model() as model_t:
    #     pm.glm.GLM.from_formula('target ~ one + two + three + four', data_c)
    #     trace = pm.sample(3000, cores=2)  # draw 3000 posterior samples using NUTS sampling
    # joblib.dump(trace, 'trace_normal_model_c.pkl')

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
    df['target_missing'] = norm.rvs(df['mu'], df['sigma'])
    return df

def outcomes_create(df):
    df['y_control'] = df.apply(lambda x: x['target'] if x['treatment'] == 0 else x['target_missing'], axis=1)
    df['y_treatment'] = df.apply(lambda x: x['target'] if x['treatment'] == 1 else x['target_missing'], axis=1)
    return df

def treatment_effect_calc(df):
    return (df['y_treatment'] - df['y_control']).mean(), (df['y_treatment'] - df['y_control']).std()


if __name__ == '__main__':
    # data_create()

    # data = joblib.load('data_normal.pkl')
    # data = pd.read_csv('radon.csv')[['county', 'log_radon', 'floor', 'county_code']]

    np.random.seed(seed=2)
    N = 100
    J = 4
    df = pd.DataFrame(np.random.uniform(-1, 1, size=(N, 3)), columns=['one', 'two', 'three'])
    df['stratum'] = np.random.choice(range(J), size=(N, 1))
    df['treatment'] = np.random.choice([0, 1], size=(N, 1))
    df['y'] = np.random.uniform(0, 1, size=(N, 1))
    data = df

    model_parameter_distribution_estimate(data)

    sys.exit()


    trace_t = joblib.load('trace_normal_model_t.pkl')
    trace_c = joblib.load('trace_normal_model_c.pkl')
    treatment = posterior_predicted_distribution(trace_c, data.query('treatment == 1')).pipe(missing_draws)
    control = posterior_predicted_distribution(trace_t, data.query('treatment == 0')).pipe(missing_draws)

    tau = treatment.append(control).\
        pipe(outcomes_create).\
        pipe(treatment_effect_calc)

    print(tau)
