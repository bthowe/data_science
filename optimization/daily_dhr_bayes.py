import sys
import numpy as np


def draw(mean, sd):
    return np.random.normal(mean, sd)


def likelihood(draw):
    likelihood_dict = {}
    for sides in [3, 4, 6, 8, 20]:
        if draw > sides:
            likelihood_dict[sides] = 0
        else:
            likelihood_dict[sides] = 1 / sides
    
    return likelihood_dict


def _normalize(posterior_dict):
    s = np.sum([posterior_dict[sides] for sides in [3, 4, 6, 8, 20]])
    return {sides: posterior_dict[sides] / s for sides in [3, 4, 6, 8, 20]}


def posterior(likelihood_dict, prior_dict):
    posterior_dict = {}
    for sides in [3, 4, 6, 8, 20]:
        posterior_dict[sides] = likelihood_dict[sides] * prior_dict[sides]
    return _normalize(posterior_dict)


def main():
    """
    Goal: distribution over the mean and sd parameters

    Initial knowledge: reasonable means?
    
    Assume a flat prior intially.
    """
    mean = np.random.uniform(0, 100)
    sd = np.random.uniform(0, 10)
    print(mean, sd)
    
    # flat prior
    prior_mean = 1  # what do I do here? I know this needs to be a continuous uniform over the range of reasonable values
    prior_sd = 1


    sys.exit()

    again = 'y'
    while again == 'y':
        d = draw(mean, sd)
        print(f'Draw: {d}')
        
        l = likelihood(d)
        p = posterior(l, prior)
        
        print(f'Current posterior: {p}')
        print(f'Keep going? ')
        again = input()
        if again == 'y':
            prior = p
    
    print(sides, p)


if __name__ == '__main__':
    main()
