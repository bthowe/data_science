import sys
import numpy as np
import pymc3 as pm
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def poisson_model():
    with pm.Model() as disaster_model:

        switchpoint = pm.DiscreteUniform('switchpoint', lower=year.min(), upper=year.max(), testval=1900)

        # Priors for pre- and post-switch rates number of disasters
        early_rate = pm.Exponential('early_rate', 1)
        late_rate = pm.Exponential('late_rate', 1)

        # Allocate appropriate Poisson rates to years before and after current
        rate = pm.math.switch(switchpoint >= year, early_rate, late_rate)

        disasters = pm.Poisson('disasters', rate, observed=disaster_data)

        trace = pm.sample(10000)

        pm.traceplot(trace)

    plt.show()

def binom_model(df):
    # todo: make sure this works ok
    with pm.Model() as disaster_model:
        switchpoint = pm.DiscreteUniform('switchpoint', lower=df['t'].min(), upper=df['t'].max())

        # Priors for pre- and post-switch probability of "yes"...is there a better prior?
        early_rate = pm.Beta('early_rate', 1, 1)
        late_rate = pm.Beta('late_rate', 1, 1)

        # Allocate appropriate probabilities to periods before and after current
        p = pm.math.switch(switchpoint >= df['t'].values, early_rate, late_rate)

        p = pm.Deterministic('p', p)

        successes = pm.Binomial('successes', n=df['n'].values, p=p, observed=df['category'].values)

        trace = pm.sample(10000)

        pm.traceplot(trace)

        plt.show()

def uniform_model(df):
    """
    The switchpoint is modeled using a Discrete Uniform distribution.
    The observed data is modeled using the Normal distribution (likelihood).
    The priors are each assumed to be exponentially distributed.
    """
    alpha = 1.0 / df['score'].mean()
    beta = 1.0 / df['score'].std()

    t = df['t_encoded'].values

    with pm.Model() as model:
        switchpoint = pm.DiscreteUniform("switchpoint", lower=df['t_encoded'].min(), upper=df['t_encoded'].max())
        mu_1 = pm.Exponential("mu_1", alpha)
        mu_2 = pm.Exponential("mu_2", alpha)
        sd_1 = pm.Exponential("sd_1", beta)
        sd_2 = pm.Exponential("sd_2", beta)
        mu = pm.math.switch(switchpoint >= t, mu_1, mu_2)
        sd = pm.math.switch(switchpoint >= t, sd_1, sd_2)
        X = pm.Normal('x', mu=mu, sd=sd, observed=df['score'].values)
        trace = pm.sample(20000)

    pm.traceplot(trace[1000:], varnames=['switchpoint', 'mu_1', 'mu_2', 'sd_1', 'sd_2'])
    plt.show()

def time_encode(df):
    df['t_encoded'] = LabelEncoder().fit_transform(df['t'])
    df.drop('t', 1, inplace=True)
    return df

def data_generate():
    mu1 = .55
    sd1 = .1
    mu2 = .50
    sd2 = .1
    df = pd.DataFrame(norm.rvs(mu1, sd1, size=(50, 1)), columns=['score'])
    df['t_encoded'] = np.random.choice(range(10), size=(50, 1))
    df2 = pd.DataFrame(norm.rvs(mu2, sd2, size=(50, 1)), columns=['score'])
    df2['t_encoded'] = np.random.choice(range(10, 20), size=(50, 1))
    return df.append(df2).sort_values('t_encoded')

if __name__ == '__main__':
    # todo: get new candidate berke scores, merge them to the ci, and then score them all.
    # todo: perform a switch point analysis on the scores.

    df = data_generate()

    # df = pd.read_csv('/Users/travis.howe/Downloads/berke_scores.csv')[['score', 'Completed On']].\
    #     rename(columns={'Completed On': 't'}). \
    #     pipe(time_encode)

    # binom_model(df)
    uniform_model(df)

# change point in time from uniform distribution
# two binomial distribution (before and after change)

