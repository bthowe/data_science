import sys
import numpy as np
import pymc3 as pm
import pandas as pd
from scipy import stats
from sklearn.neighbors import KernelDensity
from pymc3.distributions.continuous import Interpolated

# in the command line I ran < theano-cache clear > and things started running a lot faster

class Bandit(object):
    '''
    Implements an online, learning strategy to solve the Multi-Armed Bandit problem using Bayesian analysis where
    rewards are continuous and there are no covariates (i.e., non-contextual).

    parameters:
        num_bandits: the number of arms
        samples_num: the total number of mcmc samples drawn when estimating the posterior
        burn_num: discard the first burn_num of the sample_num mcmc samples
    methods:
        _sample_posteriors(arm): random draw from the specified arm
        choose_arm(): choose arm to play according to random draws
        _from_posterior(param, samples): transforms a vector of sample draws into a distribution object that can be used in the mcmc step
        posterior_update(arm, reward): updates the posteriors of the specified arm using the value of the realized reward
    attributes:
        plays: a dictionary with the indexed arms as keys and the value corresponding to the number of times the arm has been played
        rewards: a dictionary with the indexed arms as keys and the value corresponding to the sum of the rewards the arm has received
        trace: a dictionary with the indexed arms as keys and the value corresponding to the arm's posterior distribution

        lb: lower bound for the uninformative prior on the mean
        ub: upper bound for the uninformative prior on the mean
        sigma: standard deviation for the prior on the standard deviation
    '''

    def __init__(self, num_bandits, samples_num, burn_num):
        self.num_bandits = num_bandits
        self.samples_num = samples_num
        self.burn_num = burn_num
        self.plays = {i: 0 for i in xrange(num_bandits)}
        self.rewards = {i: 0 for i in xrange(num_bandits)}

        self.prior_lb = 100
        self.prior_ub = 400
        self.prior_sigma = 40

        self.trace = {i: {'mu': np.random.uniform(self.prior_lb, self.prior_ub, size=(500, 1)),
                          'sigma': stats.halfnorm(scale=self.prior_sigma).rvs(size=500)} for i in xrange(num_bandits)}

        self.draws = None

    def _sample_posteriors(self, arm):
        kde = KernelDensity()
        kde.fit(pd.DataFrame(self.trace[arm]['mu'][-(self.samples_num - self.burn_num):]))
        return float(kde.sample())

    def choose_arm(self, set_of_arms):
        self.draws = [self._sample_posteriors(arm) for arm in set_of_arms]
        return set_of_arms[np.argmax(self.draws)]

    def _from_posterior(self, param, samples):
        smin, smax = np.min(samples), np.max(samples)
        width = smax - smin
        x = np.linspace(smin, smax, 100)
        y = stats.gaussian_kde(samples)(x)

        # what was never sampled should have a small probability but not 0,
        # so we'll extend the domain and use linear approximation of density on it
        if param == 'sigma':
            x = np.concatenate([[max(0, x[0] - 3 * width)], x, [x[-1] + 3 * width]])
        else:
            x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
        y = np.concatenate([[0], y, [0]])
        return Interpolated(param, x, y)

    # def _from_posterior(self, param, samples):
    #     class FromPosterior(Continuous):
    #         def __init__(self, *args, **kwargs):
    #             self.logp = logp
    #             super(FromPosterior, self).__init__(*args, **kwargs)
    #
    #     smin, smax = np.min(samples, axis=0), np.max(samples, axis=0)
    #     x = np.linspace(smin, smax, 100)
    #     y = stats.gaussian_kde(samples)(x)
    #     y0 = np.min(y) / 10
    #
    #     @as_op(itypes=[tt.dscalar], otypes=[tt.dscalar])
    #     def logp(value):
    #         return np.array(np.log(np.interp(value, x, y, left=y0, right=y0)))
    #
    #     return FromPosterior(param, testval=np.median(samples))

    def posterior_update(self, arm, reward):
        if self.plays[arm] == 0:
            self.basic_model = pm.Model()
            with self.basic_model:
                mu = pm.Uniform('mu', lower=self.prior_lb, upper=self.prior_ub)
                sigma = pm.HalfNormal('sigma', sd=self.prior_sigma)
                Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=reward)

                # step = pm.Metropolis()
                # trace = pm.sample(self.samples_num, step=step)
                trace = pm.sample(self.samples_num)

            self.trace[arm] = trace

        else:
            self.basic_model = pm.Model()
            with self.basic_model:
                mu = self._from_posterior('mu', self.trace[arm]['mu'][-(self.samples_num - self.burn_num):])

                sigma = self._from_posterior('sigma', self.trace[arm]['sigma'][-(self.samples_num - self.burn_num):])
                Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=reward)

                # step = pm.Metropolis()
                # trace = pm.sample(self.samples_num, step=step)
                trace = pm.sample(self.samples_num)

            self.trace[arm] = trace

        self.plays[arm] += 1
        self.rewards[arm] += reward
