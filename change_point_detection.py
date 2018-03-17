# change point in time from uniform distribution
# two binomial distribution (before and after change)

import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt


def scratch():
    disaster_data = np.ma.masked_values([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                                3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                                2, 2, 3, 4, 2, 1, 3, -999, 2, 1, 1, 1, 1, 3, 0, 0,
                                1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                                0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                                3, 3, 1, -999, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                                0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1], value=-999)
    year = np.arange(1851, 1962)

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

if __name__ == '__main__':
    # todo: get new candidate berke scores, merge them to the ci, and then score them all.
    # todo: perform a switch point analysis on the scores.

    df = []

    binom_model(df)

