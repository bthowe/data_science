import sys
import numpy as np
from itertools import product


"""

multiarm bandit: exploit the info I have, explore other times.

What is the data generating process? During my time in it, how many unique individuals, not including myself, did I see

What levers could I actually pull? I mean, can I wait thirty minutes? Probably yes, but I probably wouldn't want to.
Fifteen minutes is another story. I could break it up into three or five minute increments, and these could be my
strategies. But thirty minutes would be better.
"""

restroom_data = [
    {'timestamp': '2024-10-03 13:02', 'unique_persons': 1, 'line': 0},
    {'timestamp': '2024-10-03 13:45', 'unique_persons': 0, 'line': 0},
    {'timestamp': '2024-10-08 07:52', 'unique_persons': 0, 'line': 0},
    {'timestamp': '2024-10-08 08:28', 'unique_persons': 0, 'line': 0},
    {'timestamp': '2024-10-08 09:23', 'unique_persons': 1, 'line': 0},
    {'timestamp': '2024-10-08 10:50', 'unique_persons': 1, 'line': 0},
    {'timestamp': '2024-10-10 08:00', 'unique_persons': 0, 'line': 0},
    {'timestamp': '2024-10-10 08:39', 'unique_persons': 0, 'line': 0},
    {'timestamp': '2024-10-10 09:14', 'unique_persons': 0, 'line': 0},
    {'timestamp': '2024-10-10 09:56', 'unique_persons': 1, 'line': 0},
    {'timestamp': '2024-10-10 10:21', 'unique_persons': 1, 'line': 0},
    {'timestamp': '2024-10-10 13:29', 'unique_persons': 1, 'line': 0},
    {'timestamp': '2024-10-15 08:15', 'unique_persons': 0, 'line': 0},
    {'timestamp': '2024-10-15 10:51', 'unique_persons': 1, 'line': 0},
    {'timestamp': '2024-10-15 11:52', 'unique_persons': 0, 'line': 0},
    {'timestamp': '2024-10-15 12:55', 'unique_persons': 3, 'line': 0},
    {'timestamp': '2024-10-17 07:56', 'unique_persons': 0, 'line': 0},
    {'timestamp': '2024-10-17 08:48', 'unique_persons': 0, 'line': 0},
    {'timestamp': '2024-10-17 11:55', 'unique_persons': 1, 'line': 0},
    {'timestamp': '2024-10-17 13:03', 'unique_persons': 2, 'line': 0},
    {'timestamp': '2024-10-24 07:25', 'unique_persons': 0, 'line': 0},
    {'timestamp': '2024-10-24 08:09', 'unique_persons': 0, 'line': 0},
    {'timestamp': '2024-10-24 09:10', 'unique_persons': 0, 'line': 0},
    {'timestamp': '2024-10-24 11:28', 'unique_persons': 0, 'line': 0},
    {'timestamp': '2024-11-05 08:02', 'unique_persons': 1, 'line': 0},
    {'timestamp': '2024-11-05 08:52', 'unique_persons': 1, 'line': 0},
    {'timestamp': '2024-11-05 09:51', 'unique_persons': 1, 'line': 0},
    {'timestamp': '2024-11-07 07:27', 'unique_persons': 0, 'line': 0},
    {'timestamp': '2024-11-07 09:00', 'unique_persons': 0, 'line': 0},
    {'timestamp': '2024-11-07 09:55', 'unique_persons': 1, 'line': 0},
    {'timestamp': '2024-11-07 10:56', 'unique_persons': 2, 'line': 0},
    {'timestamp': '2024-11-07 12:24', 'unique_persons': 0, 'line': 0},
    {'timestamp': '2024-11-12 07:37', 'unique_persons': 1, 'line': 0},
    {'timestamp': '2024-11-12 08:38', 'unique_persons': 0, 'line': 0},
    {'timestamp': '2024-11-12 11:00', 'unique_persons': 2, 'line': 0},
    {'timestamp': '2024-11-12 12:00', 'unique_persons': 2, 'line': 0},
    {'timestamp': '2024-11-21 07:26', 'unique_persons': 0, 'line': 0},
    {'timestamp': '2024-11-21 08:02', 'unique_persons': 0, 'line': 0},
    {'timestamp': '2024-11-21 10:25', 'unique_persons': 2, 'line': 0},
]


class Bandit:
    def __init__(self, arms_initial_dist_dict, exploit=0.8):
        self.exploit = exploit
        self.arms = arms_initial_dist_dict.keys()
        self.params_space = {arm: arms_initial_dist_dict[arm]['param_space'] for arm in self.arms}
        self.priors = {arm: arms_initial_dist_dict[arm]['initial_prior'] for arm in self.arms}
        
        self.posteriors = self.priors
        self.maps = self._maps_update()
        self.draws = []
    
    def choose_arm(self):
        """
        todo: the rule here is poor. Convenergence might be very slow. Perhaps adding something like  a burn in period might be good.
        """
        if np.random.uniform() < self.exploit:
            if self.draws:
                maps = {arm: self.maps[arm]['prob'] for arm in self.arms}
                return 'exploit', max(maps, key=maps.get)
            else:
                return 'exploit', np.random.choice(list(self.arms))
        else:
            return 'explore', np.random.choice(list(self.arms))
    
    def likelihood(self, arm, observation):
        likelihood_dict = {}
        for bounds in self.params_space[arm]:
            if observation > bounds[1]:
                likelihood_dict[bounds] = 0
            elif observation < bounds[0]:
                likelihood_dict[bounds] = 0
            else:
                likelihood_dict[bounds] = 1 / (bounds[1] + 1)
        return likelihood_dict
    
    def _maps_update(self):
        return {arm: {'params': max(self.posteriors[arm], key=self.posteriors[arm].get), 'prob': max(self.posteriors[arm].values())} for arm in self.arms}  # todo: what if there are ties?
    
    def posterior_update(self, arm, observation, exploit):
        self.draws.append((arm, observation, exploit))
        
        l = self.likelihood(arm, observation)
        p = self.priors[arm]
        
        posterior = {}
        posterior_sum = 0
        for bounds in self.params_space[arm]:
            posterior[bounds] = l[bounds] * p[bounds]
            posterior_sum += l[bounds] * p[bounds]
        
        self.posteriors[arm] = {bounds: posterior[bounds] / posterior_sum for bounds in self.params_space[arm]}
        self.priors[arm] = self.posteriors[arm]
        
        self.maps = self._maps_update()


def param_space(lower, upper):
    param_values = range(lower, upper + 1)
    return [val for val in product(param_values, param_values) if val[0] < val[1]]


def draw(arm):
    if arm == 'a':
        return np.random.randint(0, 11)
    elif arm == 'b':
        return np.random.randint(3, 16)
    elif arm == 'c':
        return np.random.randint(6, 21)
    return np.random.randint(7, 25)


def main():
    """
    The reason this gets complicated is due to the additional arms. For simplicity, do this for one arm initially and
    generalize by allowing for multiple.
    
    1. Assumption concerning underlying distribution of the draw: discrete uniform with minimum value of a and maximum
    of b.
    2. Menu of parameter values
    """
    param_space_a = param_space(0, 20)
    arms_initial_dist_dict = {
        'a': {'param_space': param_space_a, 'initial_prior': dict(zip(param_space_a, [1 / len(param_space_a)] * len(param_space_a)))},
        'b': {'param_space': param_space_a, 'initial_prior': dict(zip(param_space_a, [1 / len(param_space_a)] * len(param_space_a)))}
    }
    b = Bandit(arms_initial_dist_dict)
    while True:
        arm = b.choose_arm()
        obs = draw(arm[1])  # function of my choice, and exogenous
        b.posterior_update(arm[1], obs, arm[0])
        print(b.draws)
        print(b.posteriors)
        print(b.maps)
        input()


if __name__ == '__main__':
    main()
