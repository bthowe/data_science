import sys
import math
import numpy as np
from itertools import product
from collections import Counter
import matplotlib.pyplot as plt


def _monte_carlo_1_1(N):
    n_clubs_lst = []
    for _ in range(N):
        deck_o_suits = [0] * 13 + [1] * 13 + [2] * 13 + [3] * 13
        team_hands = np.random.permutation(deck_o_suits)[:26]
        n_clubs_lst.append(np.sum(np.where(team_hands == 0, 1, 0)))
    return n_clubs_lst


def problem_1_1():
    """
    Let's call suit 0---out of the four nominal suits 0, 1, 2, and 3---the clubs.
    
    ***One of the issues I encountered was the random nature of having all or none of the clubs, obviously. Instead of
    simply comparing the numbers, I need to compare distributions. This is obvious, but tripped me up. Specifically,
    the number of clubs in two hands is random, but also whether the number of hands with zero clubs minus the number
    of hands with thirteen clubs is also random.
    
    It is a kind of meta-probability given I'm interested in the relative frequency of occurences of two values of the
    primary random variable. I could look at the distribution of the difference and ask, "Is the mean of this
    distribution zero?" I do this in the function problem_1_1_redux, statistical test included.
    """
    
    # Using a Monte Carlo
    N = 1_000_000
    clubs_in_team_hand_distribution = np.array(_monte_carlo_1_1(N))

    # This shows the distribution of clubs in a teams hand
    counter_object = Counter(clubs_in_team_hand_distribution)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(counter_object.keys(), counter_object.values())
    plt.show()
        
    
    # Solving analytically
    print(1_000_000 * ((math.comb(13, 13) * math.comb(39, 13)) / math.comb(52, 26)))
    print(1_000_000 * ((math.comb(13, 0) * math.comb(39, 26)) / math.comb(52, 26)))


def problem_1_1_redux():
    # todo: vectorize so this doesn't take forever; also, difference of means test.
    N_meta = 1_000
    club_count_diff = []
    for _ in range(N_meta):
        N = 10_000
        clubs_in_team_hand_distribution = np.array(_monte_carlo_1_1(N))
        club_count_diff.append(np.sum(np.where(clubs_in_team_hand_distribution == 0, 1, 0)) - np.sum(np.where(clubs_in_team_hand_distribution == 13, 1, 0)))

    counter_object = Counter(club_count_diff)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(counter_object.keys(), counter_object.values())
    plt.show()

    
def problem_1_2():
    """
    Logically, three cannot be correct and one incorrect given there are three degrees of freedom: if you pin down three
    the fourth is likewise pinned. Thus, the probability is zero.
    """
    print('The probability is zero.')


def problem_1_3():
    """
    The answer depends on what it means to 'face up.' If this is interepreted as 'not facing down' then 4/5. If this is
     interepreted as the opposite direction as face down, then zero.
    """
    print('The probability is 4/5...or zero')


def _draw(p, q):
    box = [0] * p + [1] * q
    return np.random.choice(box, size=2, replace=False)

def _box_process(p, q):
    n_balls = p + q
    while n_balls > 1:
        draw = _draw(p, q)
        if np.sum(draw) == 0:
            p = p - 2
            q += 1
        elif np.sum(draw) == 2:
            q = q - 1
        else:
            q = q - 1
        n_balls = p + q
    return p, q
    
    

def problem_1_4():
    """
    It seems the answer is that the final ball will be black with probability equal to 1 if the initial number of white
    balls is even, and white with probability 1 if the initial number of white balls is odd.
    
    Proof (kind of): You can never remove only one white ball, always exactly two. This implies that if p is odd, the
    number of final white balls will be odd, and if p is initially even, the number of final white ball will be even.
    Since one (and only one) ball is removed every round, there will only be one ball left...eventually. This means, the
    final number of white balls is either zero or 1, the former if the initial number if even and one otherwise.
    """
    ## White balls
    p = 15
    
    ## black balls
    q = 13
    
    white_prob = 0
    black_prob = 0

    N = 10_000
    for _ in range(N):
        p_final, q_final = _box_process(p, q)
        if p_final == 1:
            white_prob += 1
        if q_final == 1:
            black_prob += 1
        if p_final + q_final != 1:
            print(f'problem: {p_final}, {q_final}')
            sys.exit()
    print(white_prob / N, black_prob / N)


def problem_1_5():
    """
    They all seem to work, ie the probability is 1. But why?
    """
    
    N = 100_000
    div_count = 0
    for _ in range(N):
        perm = np.random.permutation(np.arange(10))
        a = int(f'5{perm[0]}383{perm[1]}8{perm[2]}2{perm[3]}936{perm[4]}5{perm[5]}8{perm[6]}203{perm[7]}9{perm[8]}3{perm[9]}76')
        if a % 396 == 0:
            div_count += 1
    print(f'Probability of divisibility by 396: {div_count / N}')
    
    
def problem_1_6():
    """
    Analytically, P(2 or 5) = 1/3. So P(2 or 5, first die) + P(2 or 5, second die) - P(2 or 5, first and second die) =
    1/3 + 1/3 - (1/6*1/6)*4 = 12/36 + 12/36 - 4/36 = 20/36.
    """
    die = range(1, 7)
    outcomes_lst = product(die, die)
    
    N_outcomes = 0
    successful_outcomes = 0
    for outcome in outcomes_lst:
        N_outcomes += 1
        if (2 in outcome) or (5 in outcome):
            successful_outcomes += 1
    
    print(successful_outcomes / N_outcomes)


def problem_1_7():
    """
    We're interested in comparing the probabilities associated with living conditional on contracting the disease and
    taking medicine A relative to B. That is, is P(live | disease, do(medicine A)) > P(live | disease, do(medicine B)).
    The important thing to recognize is that people get better on their own. That is, what is the probability these 11
    people would have healed anyway? This is the importance of the phrase regarding the drugs not having been tested:
    the causal diagrams look different. Specifically, getting the disease is a confounding variable, influencing both
    survival AND getting the vaccine. What is the probability the results (3/3 and 7/8) are due to chance alone?
    
    P(recovery | do(A)) =
        P(recovery | A=1, better_on_own=1)P(better_on_own=1) + P(recovery | A=1, better_on_own=0)P(better_on_own=0) =
        1 * 0.5 + P(recovery | A=1, better_on_own=0) * 0.5
    
    and
    
    P(recovery | do(B)) =
        P(recovery | B=1, better_on_own=1)P(better_on_own=1) + P(recovery | B=1, better_on_own=0)P(better_on_own=0) =
        1 * 0.5 + P(recovery | B=1, better_on_own=0) * 0.5
    
    So what we ultimately care about is how P(recovery | A=1, better_on_own=0) compares to P(recovery | B=1, better_on_own=0).
    This can't be point identified using the data because we don't have data for the variable 'better_on_own', and
    logically could be any value on the interval [0, 1].
    
    Nevertheless,
    
    
    
   
    1/2 ** 3 = 1/8
    8 * (1/2 ** 7) = 1/32 (using the binomial pdf)
    
    
    
    
    """
    # todo: understand how to deconstruct the probabilities conditioning on do()
    N = 10_000
    
    disease = 0
    heal_sans_meds = 0
    for _ in range(N):
        if np.random.uniform() > 0.5:
            disease += 1
            if np.random.uniform() > 0.5:
                heal_sans_meds += 1
            
                
            
        sys.exit()


def problem_1_8():
    """
    Analytically: The probability a > c and d > b is 0.25, and a < c and d < b is likewise 0.25. Thus, the probability
    is 0.5.
    
    """
    N = 1_000_000
    
    first_quad = 0
    for _ in range(N):
        urn = np.random.choice(range(1, 2019), 4)
        a = urn[0]
        b = urn[1]
        c = urn[2]
        d = urn[3]
        
        if (a / (a * d - b * c)) > (c / (a * d - b * c)) and (d / (a * d - b * c)) > (b / (a * d - b * c)):
            first_quad += 1
    print(first_quad / N)
    

def problem_1_9():
    """
    The probability any of the six numbers are in any of the boxes is obviously 1/6. The probability that the inequality
    holds given a 1 is in the second box is 1, 4/5 given a 2 is in the second box, 3/5, etc. Thus, the desired
    probability is 1/6 * (1 + 4/5 + 3/5 + 2/5 + 1/5 + 0) = 1/2. The logic is not dependent on the location of the
    inequality, meaning the probabilities are the same.
    """
    N = 10_000
    ineq_holds = 0
    i = 3
    for _ in range(N):
        perm = np.random.permutation(range(1, 7))
        if perm[i - 1] < perm[i]:
            ineq_holds += 1
    print(ineq_holds / N)


def binary_generator(case):
    if case == 1:  # 1: 10 up, 12 down 2 up
        return [1] * 10, list(np.random.permutation([-1] * 12 + [1] * 2))
    elif case == 2:   # 1: 11 up down 1, 11 down 1 up
        return list(np.random.permutation([1] * 11 + [-1])), list(np.random.permutation([-1] * 11 + [1]))
    return list(np.random.permutation([1] * 12 + [-1] * 2)), [-1] * 10


def problem_1_10():
    """
    The key for the analytic solution is noticing that coefficient curves have to cross, and more specifically, it must
    be true that the first number is one more than the second. This implies integer roots, since c_1 = r_1 + 1, and
    c_2 = r_1 * 1.
    
    The Monte Carlo solution below was kind of a waste of time, but it simultaneously wasn't because the solution involved:
    1. how do I randomly construct paths (in this case I created random permutations of the change in state based on
    known characteristics of the paths.
    2. the various numpy functions I used: np.any, np.all, np.root, np.iscomplex, np.equal, np.random.uniform, np.random.permutation
    3. using np.mod(x, 1) to check whether a number, x, is an integer. Remember, np.mod(x, 2) checks odd/even, and
    np.mod(x, 1) checks integer/otherwise
    """
    N_total = 0
    integer_roots_count_total = 0
    for case in [1, 2, 3]:
        print(case)
        N = 10_000
        integer_roots_count = 0
        for _ in range(N):
            print('new\n')
            start_1 = 10
            start_2 = 20
            integer_roots_flag = 0
            coef_1_change, coef_2_change = binary_generator(case)
            while (start_1 != 20) or (start_2 != 10):
                list_choice = np.random.uniform()
                if (list_choice > 0.5) and coef_1_change:  # 1 has to be chosen and non-empty
                    start_1 = start_1 + coef_1_change.pop()
                elif (list_choice <= 0.5 and coef_2_change):  # 2 has to be chosen and non-empty
                    start_2 = start_2 + coef_2_change.pop()
                else:  # if chosen but empty then try again
                    continue
                
                roots = np.roots([1, start_1, start_2])
                
                if np.any(np.iscomplex(roots)):
                    continue
                if np.all(np.equal(np.mod(roots, 1), 0)):
                    integer_roots_flag = 1
                    break
            if integer_roots_flag ==0:
                sys.exit()
    
            if integer_roots_flag == 1:
                integer_roots_count += 1
        
        print(integer_roots_count / N)
        
        N_total += N
        integer_roots_count_total += integer_roots_count
    
    print(integer_roots_count_total / N_total)


def problem_1_11():
    """
    Yes. When there is a flaw in probabilistic thinking, first check whether an unconditional is being confused with
    a conditional probability, or the opposite. In this case, assuming the probability of the first award is as stated,
    an unconditional probability, the second is conditional on having won already. Seems to boil down to the cardinality
    of the choice set for receiving the award. It is neither the number of people on the earth (obviously...I have no
    chance, for example) nor is the number of previous winners (again, obviously).
    
    Further, what if we don't assume independence, so winning once influences whether you win again, which is probably
    true for an award like this (previous winners are less likely to be chosen again, regardless of resume...but there
    could be counterveiling hypotheses too, such as it was a weak year, they learned how to impress the committee, etc.).
    In the least, a previous winner would have less research to be considered for the award given a portion of his body
    of research was already awarded once. I'm actually of the naive opinion that winning the second time might be more
    impressive.
    """
    pass
    
def problem_1_12():
    """
    This again is a problem of mistaking conditional with unconditional. If you know (i.e., conditioning on) two heads
    have already been rolled, then the probability of the third head is 1/2. If you have no knowledge of the outcomes
    of any of the coins, then there are eight possible outcomes (not two), two of which involve the three coins showing
    the same face (all heads or all tails).
    
    Ultimately, the question is one of the information set.
    
    Ah, after reading the solution I can see why this more of perplexity: namely, if you look at the eight possible
    outcomes, all have at least two of the same face.
    """
    pass


def main():
    # problem_1_1()
    # problem_1_1_redux()
    # problem_1_4()
    # problem_1_5()
    # problem_1_6()
    # problem_1_7()
    # problem_1_8()
    # problem_1_9()
    # problem_1_10()
    problem_1_11()


if __name__ == '__main__':
    main()
