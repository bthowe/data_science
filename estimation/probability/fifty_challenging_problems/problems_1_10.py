import sys
import math
import primefac
import numpy as np
from itertools import product
from collections import Counter
import matplotlib.pyplot as plt




def _monte_carlo(f, N=10_000):
    count = 0
    for _ in range(N):
        if f():
            count += 1
    return count / N
    


def _p1(n_r, n_b, black_even):
    p = n_r / (n_r + n_b) * (n_r - 1) / (n_r + n_b - 1)
    if black_even:
        if n_b % 2 == 1:
            return False
        else:
            return p == 0.5
    else:
        return p == 0.5
    

def problem_1(black_even=False):
    """
    Sock drawer problem: There must be at least two red socks otherwise the probability of drawing two would be zero.
    The number of reds and blacks must satisfy: 1/2 = n_r / (n_r + n_b) * (n_r - 1) / (n_r + n_b - 1)
    
    The smallest numbers are 3 and 1, respectively, yielding the probability calculation 3/4 * 2/3 = 6/12.
    If black has to be even, then the smallest n_r and n_b can be is 15 and 6.
    """
    
    for n_r, n_b in product(range(1, 21), range(1, 21)):
        flag = _p1(n_r, n_b, black_even)
        if flag:
            print(f'Number of red socks: {n_r}, Number of black socks: {n_b}')
            break
    
    
def problem_2():
    """
    This is Cut the Knot chapter 4 problem 1.
    """
    pass


def problem_3():
    """
    The probability of getting the correct decision for the three-man jury is
        p * p * 1 + p * (1-p) * 1/2 + (1-p) * p * 1/2 =
        p * (p + (1-p)*1/2 + (1-p)*1/2) =
        p
    Let's do this numerically.
    """
    N = 100_000
    count = 0
    
    p = .25
    for _ in range(N):
        vote = list(np.random.binomial(1, p, 2)) + list(np.random.binomial(1, 0.5, 1))
        if np.sum(vote) > 1:
            count += 1
    print(count / N)
    

def problem_4():
    """
    np.random.randint for discrete uniform
    """
    N = 1_000_000
    
    first_occurence = 0
    for _ in range(N):
        first_occurence += np.where(np.random.randint(1, 7, size=100) == 6)[0][0] + 1
    print(first_occurence / N)


def problem_5():
    """
    Thinking about where the center of the coin can land, it must be greater than 3/8" from each side, meaning the
    area of the square that would earn the $0.04 return is 1/16, in each square. This is the probability, regardless of
    the number of squares, assuming the coin lands randomly on the table and the area of the lines is negligible (which
    it isn't). So this probability, 1/16 is an upper bound.
    
    TODO: How would I do this numerically? Flagged as interesting problem.
    """
    pass


def problem_6():
    """
    For a $1 stake, the expected loss is about $0.08 (specifically, a loss of 7.84%). The expected loss is about 39
    cents for a stake of $5.
    """
    N = 10_000_000
    winnings_lst = []
    
    stake = 1
    number = 1
    for _ in range(N):
        rolls = np.random.randint(1, 7, size=3)
        winnings = stake * np.sum(np.where(rolls == number, 1, 0))
        winnings_lst.append(-stake if winnings == 0 else winnings)
    print(np.mean(winnings_lst))
    

def problem_7a():
    """
    He is expected to be up around $2.76 at the end of each 36-round set.
    """
    N = 100_000
    winnings_lst = []
    
    p = 1 / 38
    
    for _ in range(N):
        wins = np.sum(np.random.binomial(1, p, 36))
        winnings = (wins * 35 + wins) - 36  # stake is $1
        
        if winnings >= 0:
            winnings += 20
        else:
            winnings -= 20
        
        winnings_lst.append(winnings)
    
    print(np.mean(winnings_lst))

def problem_7b():
    """
    I didn't use the binomial well above, essentially simulating 36 benoulli events. Below I use it much more
    efficiently, allowing me to remove the loop and simulate 100M sets of 36 rounds in like 5 seconds. The expected
    winnings are roughly $2.79. (1B simulatiuons took less than two minutes: 2.792924)
    """
    N = 1_000_000_000
    
    p = 1 / 38
    winnings = np.random.binomial(36, p, N) * 36 - 36
    print(np.mean(np.where(winnings >= 0, winnings + 20, winnings - 20)))
    

def problem_8():
    """
    I'm getting a zero, for 10M simulations.
    
    This brings up an interesting issue: what if I truly care about this very small probability, maybe because the \
    consequences are extreme, say. To get a precise estimate I need to run trillions (or more) of simulations given the
    answer (analytically) is a chance in like one in 160B. TOdo: this needs to be much more efficient. Flagged as interesting problem to further explore.
    """
    N = 10_000_000
    count = 0

    deck = ['h'] * 13 + ['d'] * 13 + ['c'] * 13 + ['s'] * 13
    for _ in range(N):
        shuffle = np.random.permutation(deck)
        
        player_one = shuffle[0:13]
        player_two = shuffle[13:26]
        player_three = shuffle[26:39]
        player_four = shuffle[39:52]
        
        if np.all(np.where((player_one == player_one[0]), 1, 0)) \
            or np.all(np.where((player_two == player_two[0]), 1, 0)) \
            or np.all(np.where((player_three == player_three[0]), 1, 0)) \
            or np.all(np.where((player_four == player_four[0]), 1, 0)):
            count += 1
    print(count / N, count)


def _craps_game():
    rolls = np.random.randint(1, 7, size=2)
    point = np.sum(rolls)
    
    if point in [7, 11]:
        return True
    elif point in [2, 3, 12]:
        return False
    else:
        while True:
            subsequent_point = np.sum(np.random.randint(1, 7, size=2))
            if subsequent_point == point:
                return True
            elif subsequent_point == 7:
                return False


def problem_9():
    """
    The book's answer is 0.49293, while I'm getting 0.4927806 across 10M simulations.
    
    Something that continues to trip me up is np.random.randint(low, high). I continue to forget that high should be
    maximum number + 1. Initially I put randint(1, 6) which lead to an answer around 0.488.
    """
    print(_monte_carlo(_craps_game, 10_000_000))
    

def problem_10():
    """
    For problem a, in expectation the winnings would be $5. But the amount I'm willing to actually put down in order to
    play this game differs, and is a function of my idiosyncratic internal configuration. I'd probably put $3 down.
    
    For problem b, the amount I'd put down is less since there is additional uncertainty...at the hyperparameter level.
    
    AFter consulting the answer, I'm an idiot. The additional uncertainty seems not to matter. Is it possible for the
    expected value to be greater than $5? Yes, if you collude. But the point is, if you randomize your choice, my
    friend's choice doesn't matter, given I play ONLY ONCE.
    
    [n_b / (n_b + n_w)] * 0.5 * 10 + [n_w / (n_b + n_w)] * 0.5 * 10 = 5
    
    What happens if we play repeatedly? It becomes a game with more complicated strategic aspects.
    
    After thinking through it again, I'm less sure of the book's answer. If you randomize, the expected value is still
    $5. But why does that imply I'd be willing to put down what I put down above, assuming no collusion? Essentially
    this expected value assumes equal preference across outcomes. For example, suppose the friend, so called, puts in
    only black balls. In this case there is a 0.5 chance I have NO CHANCE of winning. Might it be rational to want a
    certain minimum threshold of possiblity of winning?
    """
    
def main():
    # problem_1()
    # problem_1(black_even=True)
    # problem_3()
    # problem_4()
    # problem_6()
    # problem_7a()
    # problem_7b()
    # problem_8()
    problem_9()
    

if __name__ == '__main__':
    main()
