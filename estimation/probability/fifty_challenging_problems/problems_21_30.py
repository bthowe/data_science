import sys
import math
import joblib
import itertools
import numpy as np
import pandas as pd
from itertools import cycle
from functools import partial

import scipy.stats
from scipy.stats import binom


pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)



def _monte_carlo(f, N=10_000):
    count = 0
    for _ in range(N):
        count += f()
    return count / N


def likelihood(d, first_draw=None, second_draw_action='replace'):
    if (not first_draw) or (second_draw_action == 'replace'):
        if d == 'b':
            return [1/3, 100/201]
        return [2/3, 101/201]
    else:
        if first_draw == 'b':  # first draw was black
            if d == 'b':
                return [0, 99/200]  # probabilities associated with seeing a second black, conditional on each urn
            return [1, 101/200]  # probabilities associated with seeing a red, conditional on each urn
        else:  # first draw was red
            if d == 'b':
                return [1/2, 100/200]  # probabilities associated with seeing a black, conditional on each urn
            return [1/2, 100/200]  # probabilities associated with seeing a second red, conditional on each urn


def urn_probs(urn, first_draw_color=None, second_draw_action='replace'):
    if (not first_draw_color):  # if first draw
        if urn == 'A':
            return 2/3  # probability of red
        return 101/201
    else:  # if second draw
        if second_draw_action == 'replace':  # if replacing the first ball
            return urn_probs(urn, first_draw_color=None)  # the probability is the same as for the first draw
        else:  # if not replacing the first ball
            if urn == 'A':
                if first_draw_color == 'r':
                    return 1/2
                return 1
            else:
                if first_draw_color == 'r':
                    return 100/200
                return 101/200


def _map(urn, posterior):
    if posterior[0] == posterior[1]:
        return int(np.where(np.random.choice(['A', 'B']) == urn, 1, 0))
    return int(np.where({0: 'A', 1: 'B'}[np.argmax(posterior)] == urn, 1, 0))
    
    
def problem_21_MAP():
    """
    Methodology: Make a random draw from Choose an urn, choose a ball. Calculate the posterior distribution. Make a
    guess based on MAP. How often is this correct? See notes.md file for commentary.
    """
    N = 100_000
    outcome_lst = []
    for _ in range(N):
        urn = np.random.choice(['A', 'B'])
        p1 = urn_probs(urn)
        first_ball = {
            0: 'b',
            1: 'r'
        }[np.random.binomial(1, p1)]
        # first_ball = 'r'
        
        ## randomly choose and action
        action = np.random.choice(['replace', 'without_replacement'])
        # action = 'replace'
        
        # draw the second ball
        p2 = urn_probs(urn, first_draw_color=first_ball, second_draw_action='without_replacement')
        second_ball = {
            0: 'b',
            1: 'r'
        }[np.random.binomial(1, p2)]
        
        print(first_ball)
        print(action)
        print(second_ball)
        
        #calculate the posterior probability of urn
        # posterior = np.array([1/2, 1/2]) * np.array(likelihood(second_ball, first_draw=first_ball, second_draw_action=action))  # we see the first ball but not the urn label
        posterior = likelihood(second_ball, first_draw=first_ball, second_draw_action=action) / np.sum(likelihood(second_ball, first_draw=first_ball, second_draw_action=action))  # uniform prior, so not information
        print(posterior)
        print(urn)
        print(_map(urn, posterior))
        sys.exit()
        outcome_dict = {
            'first_ball': first_ball,
            'action': action,
            'correct': _map(urn, posterior)
        }
        outcome_lst.append(outcome_dict)
    df = pd.DataFrame(outcome_lst)
    pivot = pd.pivot_table(df, values='correct', index='first_ball', columns='action', aggfunc=np.mean)
    print(pivot)


def urn_A():
    return list(np.random.permutation(['r'] * 2 + ['b']))


def urn_B():
    return list(np.random.permutation(['r'] * 101 + ['b'] * 100))


def draw():
    urn_name = np.random.choice(['A', 'B'])
    urn = {
        'A': urn_A(),
        'B': urn_B(),
    }[urn_name]
    
    first_ball = urn.pop()

    action = np.random.choice(['replace', 'without_replacement'])
    if action == 'replace':
        urn += first_ball
        urn = list(np.random.permutation(urn))
    
    second_ball = urn.pop()
    
    return {'urn': urn_name, 'first_ball': first_ball, 'second_ball': second_ball, 'action': action}


def problem_21():
    """
    Here I don't fuss with probabilities but solve the problem by GENERATING DATA. See notes.md file for commentary.
    """
    N = 10_000_000
    df = pd.DataFrame([draw() for _ in range(N)])
    joblib.dump(df, '/Users/travis/Projects/data_science/estimation/probability/fifty_challenging_problems/problem_21.pkl')
    
    for first_ball, second_ball in itertools.product(['r', 'b'], ['r', 'b']):
        for action in ['replace', 'without_replacement']:
            df_temp = df. \
                query(f'action == "{action}"').\
                query(f'first_ball == "{first_ball}"').\
                query(f'second_ball == "{second_ball}"')
            
            denom = df_temp.shape[0]
            N_A = df_temp.query('urn == "A"').shape[0]
            N_B = df_temp.query('urn == "B"').shape[0]
            P_A =  N_A / denom
            P_B =  N_B / denom
            print(f'{first_ball}, {second_ball}, {action}: P[Urn = A] = {P_A}, P[Urn = B] = {P_B}, N = {denom}, N_A = {N_A}, N_B = {N_B}')
        print('\n')


def ballot_box(N_a, N_b):
    ballot_box = list(np.random.permutation(['a'] * N_a + ['b'] * N_b))
    a = 0
    b = 0
    while ballot_box:
        draw = ballot_box.pop()
        if draw == 'a':
            a += 1
        else:
            b += 1
        if a == b:
            return 1
    return 0

def problem_22():
    """
    Very straightforward
    """
    N_a = 10
    N_b = np.random.choice(N_a)
    
    print(_monte_carlo(partial(ballot_box, N_a, N_b), N=1_000_000))
    print(2/((N_a/N_b) + 1))


def matching_pennies(N):
    a_wins = 0
    b_wins = 0
    
    for _ in range(N):
        if np.random.uniform() > .5:
            a_wins += 1
        else:
            b_wins += 1
        if a_wins == b_wins:
            return 0
    return 1
    

def problem_23():
    """
    Most interesting aspect of this problem was the python function that calculates N choose k: math.comb(N, k).
    """
    N_games = 16
    if N_games % 2 == 0:
        print(math.comb(N_games, int(N_games / 2)) / 2 ** (N_games))
    else:
        print(math.comb(N_games - 1, int(N_games / 2)) / 2**(N_games - 1))
    print(_monte_carlo(partial(matching_pennies, N_games), N=10_000_000))


def problem_24():
    """
    We did this one in Cut the Knot.
    """
    pass


def chord_length(radius):
    central_angle = math.radians(np.random.uniform(0, 360))
    if (2 * radius * math.sin(central_angle / 2)) > radius:
        return 1
    return 0

    
def problem_25():
    """
    Methodology: (1) randomly draw an angle, (2) calculate the chord length which is a function of the radius (given)
    and the angle,
    
    My approach was the same as the answer described in solution (c).
    """
    radius = 20
    print(_monte_carlo(partial(chord_length, radius), N=1_000_000))
    
    
def duelists():
    a_arrive = np.random.choice(3600)
    a_leave = a_arrive + 300
    b_arrive = np.random.choice(3600)
    b_leave = b_arrive + 300
    
    if (a_arrive <= b_arrive <= a_leave) or (b_arrive <= a_arrive <= b_leave):
        return 1
    return 0
    
    
def problem_26():
    """
    The book uses the minute as the smallest unit of time and consequently gets 23/144 = 0.1597222222222222 as an answer.
    Using seconds instead, over 10M simulations yields a probability of 0.160057
    
    """
    print(_monte_carlo(duelists, N=10_000_000))
    
    
def minter_a(n):
    if np.sum(np.random.binomial(1, 1/n, n)) >= 1:
        return 1
    return 0

def problem_27():
    """
    Here I calculate the probability of detection.
    
    A good ratio to remember: 1/e = 0.367879...
    
    So this is, apparently the limit as n goes to infinity. Is this something I could ever pick up on doing this numerically?
    That is, the differnce between say, the estimate for n=1_000 versus 10_000?
    
    """
    # Theoretical: this is a simple manipulation of a binomial
    print(1 - binom.cdf(0, 100, 1/100))
    
    # monte carlo
    # print(_monte_carlo(partial(minter_a, 100), N=1_000_000))
    print(_monte_carlo(partial(minter_a, 100_000), N=100_000))


def counterfeit_coins_redux(n, m, r):
    false_count = 0
    for _ in range(n):
        box = list(np.random.permutation([True] * (n-m) + [False] * m))
        draw = box.pop()
        
        if not draw:
            false_count += 1
    if false_count == r:
        return 1
    return 0


def problem_28():
    """
    Because this comes up occasionally, how important is it to derive an analytical expression for the function? I have
    here a numerical function.
    """
    n = 10  # coins in a box and number of boxes from which 1 coin is drawn
    m = 2  # false coins in a box
    r = 3

    print(_monte_carlo(partial(counterfeit_coins_redux, n, m, r)))
    print(math.comb(n, r) * ((m / n) ** r) * ((1 - (m / n))**(n - r)))


avg_colonies = 3

avg_colonies = 3

avg_colonies = 3


def _problem_29_solution1(avg_colonies):
    N_plates_dim = 1000
    plates = np.zeros(shape=(N_plates_dim, N_plates_dim))
    for _ in range(N_plates_dim ** 2 * avg_colonies):
        i = np.random.randint(N_plates_dim)
        j = np.random.randint(N_plates_dim)
        
        plates[i, j] += 1
    
    print(np.sum(np.where(plates == avg_colonies, 1, 0)) / (N_plates_dim ** 2))


def _problem_29_solution2(avg_colonies, N_plates):
    plates = np.zeros(shape=N_plates)
    for i in np.random.randint(N_plates, size=N_plates * avg_colonies):
        plates[i] += 1
    print(np.sum(np.where(plates == avg_colonies, 1, 0)) / N_plates)


def problem_29():
    """
    Who knows? It depends on the underlying distribution. For example, suppose there are 10 plates, nine of which have 0
    spores. Then if the tenth plate has thirty spores, the average is three.
    
    What in the world is being assumed that allows this bloke to come up with an answer? That each of partitions on each
    plate has the same probability of having a colony, as though the spores were being uniformly sprinkled over the room.
    This is an important assumption.
    """
    avg_colonies = 3

    ## First solution using a two-dimensional array because it was more intuitive.
    # _problem_29_solution1()
    
    ## Second solution that does not use a two-dimensional array
    _problem_29_solution2(avg_colonies, 1_000_000)
    
    
def problem_30():
    """
    Here we assume a Poisson, thank you for this information.
    """
    N = 1_000_000
    print(np.sum(np.where(np.random.poisson(20, size=N) % 2 == 0, 1, 0)) / N)
    
    
def main():
    # problem_21_MAP()
    # problem_21()
    # problem_22()
    # problem_23()
    # problem_25()
    # problem_26()
    # problem_27()
    # problem_28()
    problem_29()
    # problem_30()
    

if __name__ == '__main__':
    main()
