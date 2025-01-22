import sys

import numpy as np

def draw(sides):
    return np.random.randint(1, sides + 1)


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
    Goal: distribution over the sides parameter
    
    Initial knowledge: how many sides there might be: 3, 4, 6, 8, or 20.
    """
    sides = list(np.random.permutation([3, 4, 6, 8, 20])).pop()
    prior = {
        3: 1 / 5,
        4: 1 / 5,
        6: 1 / 5,
        8: 1 / 5,
        20: 1 / 5
    }
    
    again = 'y'
    while again == 'y':
        d = draw(sides)
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
