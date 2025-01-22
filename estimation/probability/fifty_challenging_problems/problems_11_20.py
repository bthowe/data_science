import sys
import numpy as np
import pandas as pd
from itertools import cycle
from functools import partial
from scipy.stats import binom


def _monte_carlo(f, N=10_000):
    count = 0
    for _ in range(N):
        count += f()
    return count / N


def problem_11_and_12():
    """
    In 11, what someone is most likely to choose is a matter of estimation. I'd choose 1.
    
    In 12, I'd choose the statue of liberty. I also liked his suggestion of the TOP of the empire state building (I
    thought of the Chrystler building). Why? fewer people in a famous spot. He does not recommend the statue of liberty
    because it's hard to get to...I thought that is exactly a reason to choose it. Relatively few people (compared to
    somewhere like Time Square), a spot that my companion might choose because of its notoriety, and the extra work to
    get there is a point of differentiation.
    """


def problem_13():
    """
    This sounds a little like the Monte Hall goat/corvette problem.
    
    This is very unintuitive. I can see why she might think the probability is 1/2. If the person to be released was
    randomly chosen to be revealed, the probability would be 1/2.
    
    Probability that she is to be released and also revealed: 2/3 * 1/2
    Probability that she is to be released and not revealed: 2/3 * 1/2
    Probability that she is not to be released: 1/3
    
    Thus, the (unconditional) probability of being released is 2/3 * 1/2 + 2/3 * 1/2 = 2/3. Conditional on not being
    revealed, the probability of being released is [2/3 * 1/2] / [2/3] = 1/2. (divide by 2/3 because it's conditional
    on not being revealed).
    
    Nevertheless, the probability in question ISN'T as described because the revealed prisoner is not randomly chosen
    from the two that will be released. The probabilities now become
    
    Probability that she is to be released and also revealed: 2/3 * 0
    Probability that she is to be released and not revealed: 2/3 * 1
    Probability that she is not to be released: 1/3
    
    Thus, given the warden's revelation, the probability she is released is 2/3. The warden reveals no additional info
    regarding her release (not the case, though, of prisoner B's release).
    """


def coupons():
    coupon_lst = []
    
    complete_set = False
    counter = 0
    while not complete_set:
        coupon_lst.append(np.random.randint(1, 6))
        counter += 1
        
        if np.all(np.isin([1, 2, 3, 4, 5], coupon_lst)):
            complete_set = True
    return counter
    
    
def problem_14():
    """
    Good question for a Monte Carlo. I get 11.4055, which is the answer given in the solution.
    """
    print(_monte_carlo(coupons, 10_000_000))


def match_making():
    seating = list(np.random.permutation(['b'] * 8 + ['m'] * 7))
    
    prior_seat = seating.pop()
    eligible_pairs = 0
    for seat in seating:
        # print(prior_seat, seat)
        if prior_seat != seat:
            eligible_pairs += 1
        prior_seat = seat
        # print(eligible_pairs)
    return eligible_pairs


def problem_15():
    """
    Ha, can't phrase this question in this manner any more! The book's solution is 7 7/15 = 7.46666, and mine here yields
    7.467
    """
    print(_monte_carlo(match_making, N=10_00_000))
    
    
def tourny():
    seeding = np.random.permutation(range(1, 9))
    
    while len(seeding) > 2:
        seeding = [min(game) for game in zip(seeding[::2], seeding[1::2])]
    
    if 2 in seeding:
        return 1
    return 0


def problem_16():
    """
    The thing to remember is the slicing/iterating over seeding, using [::2] and [1::2].
    
    4 / 7 = 0.5714285714285714. I got 0.5714951.
    """
    print(_monte_carlo(tourny, N=10_000_000))


def knights_tourny(n):
    seeding = np.random.permutation(['k'] * (2**n - 2) + ['b'] * 2)
    while len(seeding) > 1:
        winners = []
        for game in zip(seeding[::2], seeding[1::2]):
            if ('b', 'b') == game:
                return 1
            winners.append(np.random.choice(game))
        seeding = winners
    return 0


def problem_17():
    """
    The book's solution for part (a) is 1/4, which is what I got as well. For part (b), simply change the 6 in line 114
    to 2**n - 2 and run.
    
    The big innovation here was remembering the partial module which allows me to pass a function with an argument
    into _monte_carlo without invoking it.
    
    I'm seeing 0.062376 over 1M simulations. But why is this taking so long? Going to refactor.
    """
    # print(_monte_carlo(partial(knights_tourny, n=3), N=10_000_000))
    print(_monte_carlo(partial(knights_tourny, n=5), N=1_000_000))


def problem_18():
    """
    This one is a simple use of the binomial pdf.
    """
    # analytical solution
    print(binom.pmf(50, 100, 0.5))

    # numerical solution
    N = 10_000_000
    print(np.mean(np.where(np.sum(np.where(np.random.uniform(size=(N, 100)) > 0.5, 1, 0), axis=1) == 50, 1, 0)))


def rolls(n_rolls):
    rolls = np.random.randint(1, 7, size=n_rolls)
    if np.sum(np.where(rolls == 6, 1, 0)) >= (n_rolls / 6):
        return 1
    return 0


def problem_19():
    """
    I find this outcome prima facie counterintuitive.
    
    It isn't stictly that we divide the twelve rolls, say, into two groups and
    require at least one six because if you get two in the first six you can get zero in the second. So the probability
    would be the probability of at least 1 in six rolls squared minus stuff. But the point is that it will be less
    than the initial probability because you're doing something random multiple times, in part. This explanation is very
    poor but it helps with my intuition a bit.
    """
    # empirical
    print(_monte_carlo(partial(rolls, 6), 100_000))
    print(_monte_carlo(partial(rolls, 12), 100_000))
    print(_monte_carlo(partial(rolls, 18), 100_000))
    
    # theoretical
    print(1 - binom.cdf(0, 6, 1/6))
    print(1 - binom.cdf(1, 12, 1/6))
    print(1 - binom.cdf(2, 18, 1/6))


class Duel:
    def __init__(self):
        self.duelist_hit_probs = {
            'a': 0.3,
            'b': 1,
            'c': 0.5,
        }
        
        self.current_duels = {
            'a': np.random.choice(['b', 'c']),
            'b': np.random.choice(['a', 'c']),
            'c': np.random.choice(['a', 'b'])
        }
        self.a_initial_target = self.current_duels['a']
        
        self.living_duelists_lst = ['a', 'b', 'c']

    def _shoot(self, shooter):
        if np.random.uniform() < self.duelist_hit_probs[shooter]:
            return 'hit'
        return 'miss'
    
    def _shots_fired(self, living_duelists_lst):
        hit = []
        for duelist in living_duelists_lst:
            opponent = self.current_duels[duelist]
            if self._shoot(duelist) == 'hit':
                hit.append(opponent)
        return list(set(hit))

    def round(self):
        living_duelists_lst = self.living_duelists_lst
        # print(self.current_duels)
        
        hit_duelists = self._shots_fired(living_duelists_lst)
        # print(f'hit: {hit_duelists}')
        
        for hit_duelist in hit_duelists:
            living_duelists_lst.remove(hit_duelist)
        # print(f'living: {living_duelists_lst}')
        
        self.living_duelists_lst = living_duelists_lst
        
    def duel(self):
        
        while len(self.living_duelists_lst) > 1:
            self.round()
            
            if len(self.living_duelists_lst) == 0:
                # print('all dead')
                pass
            elif len(self.living_duelists_lst) == 1:
                # print(f'{self.living_duelists_lst[0]} is the winner')
                if self.living_duelists_lst[0] == 'a':
                    return self.a_initial_target
            else:
                self.current_duels = {
                    self.living_duelists_lst[0]: self.living_duelists_lst[1],
                    self.living_duelists_lst[1]: self.living_duelists_lst[0]
                }
                

def problem_20a():
    """
    Assume duelists all fire at the same time. For 1M duelist 'a' victories, he initially fired at 'b' and 'c',
    respectively 615,420 and 384,580 times.
    
    Idiotically, initially I didn't read the question closely enough to realize that the shooting follows the sequence
    'a', 'b', 'c'.
    """
    
    a_initial_opponents = []
    while len(a_initial_opponents) < 1_000_000:
        d = Duel()
        a = d.duel()
        if a:
            a_initial_opponents.append(a)
    
    print(pd.Series(a_initial_opponents).value_counts())


class Duelb:
    def __init__(self):
        self.duelist_hit_probs = {
            'a': 0.3,
            'b': 1,
            'c': 0.5,
        }
        
        self.current_shooter = 'a'
        
        self.a_initial_action = None
        
        # self.a_initial_target = self.current_duels['a']
        
        self.living_duelists_lst = ['a', 'b', 'c']
    
    def round(self, shooter):
        possible_oponents = [opponent for opponent in self.living_duelists_lst if opponent != shooter]
        action = np.random.choice(['intentional_miss'] + possible_oponents)
        print(f'\taction: {action}')
        
        if shooter == 'a' and len(self.living_duelists_lst) == 3:
            self.a_initial_action = action
        
        if action == 'intentional_miss':
            # print('intential miss')
            pass
        elif np.random.uniform() < self.duelist_hit_probs[shooter]:
            print('\t\thit')
            self.living_duelists_lst.remove(action)
        else:
            print('\t\tmiss')
    
    def duel(self):
        
        while len(self.living_duelists_lst) > 1:
            
            print(f'current list of living duelists: {self.living_duelists_lst}')
            print(f'current shooter: {self.current_shooter}')
            
            self.round(self.current_shooter)
            
            self.living_duelists_lst.remove(self.current_shooter)
            self.living_duelists_lst.append(self.current_shooter)
            
            self.current_shooter = self.living_duelists_lst[0]
            
            if len(self.living_duelists_lst) == 1:
                print(f'{self.living_duelists_lst[0]} is the winner')
                return self.living_duelists_lst[0]


def problem_20b():
    """
    The book's solution disregards all combatants shooting into the air (really, it didn't provide enough structure...
    for example not mentioning that A shooting into the air is an option, and if an option for A then why not for B and
    C?).
    
    todo: if all shooters shoot in the air then it's over...this seems reasonable.
    """
    
    a_initial_opponents = []
    while len(a_initial_opponents) < 10_000:
        d = Duelb()
        if d.duel() == 'a':
            a_initial_opponents.append(d.a_initial_action)
        print('\n\n')
    
    print(pd.Series(a_initial_opponents).value_counts())

    
def main():
    # problem_14()
    # problem_15()
    # problem_16()
    # problem_17()
    # problem_18()
    # problem_19()
    # problem_20a()
    problem_20b()
    
    
if __name__ == '__main__':
    main()
