import sys
import numpy as np
from time import time
from scipy.stats import norm
from itertools import product

param_space = list(product(list(np.linspace(.2, 100, 500)), list(np.linspace(.2, 10, 50))))


def draw(mean, sd):
    return np.random.normal(mean, sd)


def likelihood(draw):
    return dict(zip(param_space, norm.pdf(draw, [val[0] for val in param_space], [val[1] for val in param_space])))
    

def _normalize(posterior_dict):
    s = np.sum([posterior_dict[val] for val in param_space])
    return {val: posterior_dict[val] / s for val in param_space}


def posterior(likelihood_dict, prior_dict):
    posterior_dict = {val: likelihood_dict[val] * prior_dict[val] for val in param_space}
    return _normalize(posterior_dict)


def main():
    """
    1. Make an assumption concerning the underlying distribution from which the draw originates: e.g., daily DHR values
    are normal.
    2. Make an array of possible parameter values for this distribution: e.g., the mean is between 0 and 100, the
    standard deviation is between 0 and 10, and i will consider two tenths increments of in this range
    3. Harvest a draw, i.e. a data point
    4. Calculate the likelihood: conditional on the possible combinations of the parameters (see step 2) and the assumed
    distribution (see step 1), what is the probability of the draw?
    5. The initial prior is flat
    6. Calcualte the posterior by multiplying the likelihood by the prior and then normalize.
    7. Iterate: Harvest another draw, calculate the likelihood, update the prior to be equal to the previous iterations
    posterior, and calculate the new posterior.
    8. Do this until you achieve some type of convergence.
    """
    mean = np.random.uniform(0, 100)
    sd = np.random.uniform(0, 10)
    
    # flat prior
    pr = {val: 1 / (len(list(param_space))) for val in param_space}
    
    again = 'y'
    count = 0
    while again == 'y':
        
        count += 1
        d = draw(mean, sd)
        l = likelihood(d)
        p = posterior(l, pr)
        map = max(p.values())
        print(f'Count: {count}, Draw: {d}, MAP: {map}')
        if map < 0.9:
            pr = p
        else:
            again = 'n'
    print(mean, sd, max(p, key=p.get), count)


if __name__ == '__main__':
    main()
