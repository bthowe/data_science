import sys
import constants
import numpy as np
import scipy.stats as stats

class BanditGo(object):
    def __init__(self, initial_strategy, bandit):
        self.current_arm = initial_strategy
        self.bandit = bandit

        self.results = {
            'upper_left': [0, 0],
            'upper_middle': [0, 0],
            'upper_right': [0, 0],
            'lower_left': [0, 0],
            'lower_middle': [0, 0],
            'lower_right': [0, 0]
        }
        self.N = 0

    def _outcome(self, action):
        k = np.random.choice(list(constants.p_shot.keys()), p=list(constants.p_shot.values()))  # kicker choice of strategy

        if np.random.uniform(0, 1) < constants.p_miss[k]:
            return 1
        if k == action:
            if np.random.uniform(0, 1) < constants.p_block[k]:
                return 1
        return 0

    def update(self):
        self.results[self.current_arm][0] += self._outcome(self.current_arm)
        self.results[self.current_arm][1] += 1

    def choose_arm(self):
        wins = np.array([win_trial_lst[0] for win_trial_lst in self.results.values()])
        trials = np.array([win_trial_lst[1] for win_trial_lst in self.results.values()])
        self.N += 1
        self.current_arm = self.bandit(wins, trials, self.N)


def bayesian(wins, trials, N):
    '''
    Randomly sample from a beta distribution for each bandit and pick the one
    with the largest value.
    Return the index of the winning bandit.
    '''
    return constants.strats[np.argmax(stats.beta.rvs(1 + wins, 1 + trials - wins))]


def softmax(wins, trials, N):
    '''
    Pick an bandit according to the Boltzman Distribution.
    Return the index of the winning bandit.
    '''
    tau = 3.345e-2

    mean = wins / (trials + 1)
    scaled = np.exp(mean / tau)
    probs = scaled / np.sum(scaled)
    print(probs)
    return constants.strats[np.random.choice(range(0, 6), p=probs)]  # since there are six strategies


def ucb1(wins, trials, N):
    '''
    Pick the bandit according to the UCB1 strategy.
    Return the index of the winning bandit.
    '''

    if len(trials.nonzero()[0]) < 6:
        return constants.strats[np.random.randint(0, 6)]
    else:
        means = wins / (trials + 1)
        confidence_bounds = np.sqrt((2. * np.log(N)) / trials)
        upper_confidence_bounds = means + confidence_bounds
        return constants.strats[np.argmax(upper_confidence_bounds)]


if __name__ == '__main__':
    bg = BanditGo('lower_middle', softmax)

    count = 0
    chosen_strats = []
    while count < 100000:
        bg.update()
        bg.choose_arm()
        chosen_strats.append(bg.current_arm)
        count += 1

    print('\n')
    print(bg.results)
    print(dict(zip(bg.results.keys(), [val[0] / val[1] for val in bg.results.values()])))

    print(len([strat for strat in chosen_strats[-1000:] if strat == 'lower_left']))
    print(len([strat for strat in chosen_strats[-1000:] if strat == 'lower_right']))
