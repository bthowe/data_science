import sys
import math
import primefac
import numpy as np
from itertools import product
from collections import Counter
import matplotlib.pyplot as plt


def problem_3_1():
    """
    This is not different than lining all of the balls up and saying, "What is the probability that the ith ball is
    white?" Phrased this way it is intuitive that the probability is w / (w + b). The idea of drawing balls from
    the urn leads to a non-intuitive result, stemming from the obvious fact that the number of balls in the bag are
    diminished by one after each draw. Nevertheless, we don't have to explicitly take the information set into
    consideration when constructing the probability.
    
    For the first, the probability is obviously: w / (w + b)
    
    For the second, the probability of white is (w-1) / (w-1 + b) if white was drawn initially and w / (w + b-1)
    otherwise, each occurring with probabilities w/(w + b) and b/(w + b). Thus, the probability is
    w / (w + b) * (w-1) / (w-1 + b) + b / (w + b) * w / (w + b-1) = w / (w + b).
    """
    w = 29
    b = 31
    
    N = 100_000
    w_count_draw_1 = 0
    b_count_draw_1 = 0
    w_count_draw_2 = 0
    b_count_draw_2 = 0
    for _ in range(N):
        bag = list(np.random.permutation(['w'] * w + ['b'] * b))
        draw_1 = bag.pop()
        draw_2 = bag.pop()
        
        if draw_1 == 'w':
            w_count_draw_1 += 1
        else:
            b_count_draw_1 += 1
        if draw_2 == 'w':
            w_count_draw_2 += 1
        else:
            b_count_draw_2 += 1
    print(w_count_draw_1 / N)
    print(w_count_draw_2 / N)
    print(w / (w + b))
    
    print(b_count_draw_1 / N)
    print(b_count_draw_2 / N)
    print(b / (w + b))


def problem_3_2():
    """
    These exercises really are so fun. For the numerical solution below, I use a binomial to generate the flips, and then
    compare.
    
    How would I do this analytically? Seems annoying. Below I use 1 and 2 to denote A and B respectively.
    
    P_1(1) * (P_2(2) + P_2(3) + P_2(4) + ... + P_2(21)) +
    P_1(2) * (P_2(3) + P_2(4) + ... + P_2(21)) +
    P_1(3) * (P_2(4) + ... + P_2(21)) + ... =

    P_1(1) * (P_2(2) + P_2(3) + P_2(4) + ... + P_2(20))
    P_1(2) * (P_2(3) + P_2(4) + ... + P_2(20)) +
    P_1(3) * (P_2(4) + ... + P_2(20)) +
    P_1(19) * P_2(20) + ... +
    P_2(21) * (P_1(1) + P_1(2) + P_1(3) + ... + P_1(20)) =
    
    P_1(1) * (P_2(2) + P_2(3) + P_2(4) + ... + P_2(20))
    P_1(2) * (P_2(3) + P_2(4) + ... + P_2(20)) +
    P_1(3) * (P_2(4) + ... + P_2(20)) + ... +
    P_1(19) * P_2(20) +
    P_2(21) * 1 =
    
    todo: Find another way to do this other than calculating all of these stupid proabilities
    """
    N = 10_000_000
    count = 0
    for _ in range(N):
        if np.sum(np.random.binomial(1, .5, 20)) < np.sum(np.random.binomial(1, .5, 21)):
            count += 1
    print(count / N)
    
    
def problem_3_3():
    """
    The unfortunately obvious answer is there is randomness in when he gets to the station and when either train gets to
    the station. So in spite of the timetable, the uncertainty could play out such that he doesn't take the same train
    every time, or either train an equal number of times (approximately), or some other combination of times such as
    one twice as much as the other.
    
    Upon rereading the question, I need to assume (1) his arrival is random, and (2) it is the only random component
    (i.e., the trains always arrive when they are supposed to and at a fixed interval).
    
    Think of his arrival time as uniform, say, between two values (wtg) 0 and 1. This space can be partitioned into
    segments depending on which train he'd take conditional on arriving at that time. Depending on the relative arrivals
    of each train (which is deterministic), the probability he takes train 1 could be anything strictly between 0 and 1.
    For example, if train 1 arrives a hair after train 2 every time, then the probability of taking it is effectively 0
    (epsilon). If the time between train arrivals is the same (so each train arrives 0.5 after the other), the probability
    is 0.5.
    
    I struggled with this one a bit. It is sometimes hard to parse these type of stories. The key is recognizing that
    trains arrive going in the same direction at the same interval. So, say, once every three minutes. And that trains
    going in opposite directions don't arrive at the same time.
    """
    pass


def problem_3_4():
    """
    Useful to encounter these numpy functions: np.random.choice, np.diff, and np.all.
    
    The second question is obviously 1/99. Why isn't this correct? Does this have to do with looking at the numbers or
    not after the draws? I know it's not, but in this case it seems pathological.
    """
    # todo: try this one again
    N = 10_000_000
    count = 0
    for _ in range(N):
        draw = np.random.choice(range(1, 100), size=5, replace=False)
        if np.all(np.diff(draw) > 0):
            count += 1
    print(count / N)
    
    
def problem_3_5():
    """
    The definition of coprime is having no common prime factors. The probability two randomly selected are both even is
    1/4, so the probability is less than 0.75. Initially numerically, I'm seeing a probability around 0.607925 (this is
    after 10 million trials...it's hard to believe so many cases are in fact coprime. But the correct answer is,
    apparently, 6 / (np.pi**2) = 0.6079271018540267. Bingo.
    
    
    numpy functions: np.random.randint(), np.isin (for checking whether lists contain common elements), np.any
    
    primefac.primefac to generate the prime factorization
    
    /=, *=, and -= are valid operators, similar to +=
    """
    N = 10_000_000  # 0.608344, 0.607925
    count = 0

    for _ in range(N):
        int1 = np.random.randint(1e10)
        int2 = np.random.randint(1e10)
        if not np.any(np.isin(list(set(primefac.primefac(int1))), list(set(primefac.primefac(int2))))):  # i.e, true if coprime
            count += 1
    print(count / N)
    
    
def problem_3_6():
    """
    2n >= 4 points are (uniformly) randomly chosen from [0, 1]. That is, 2n points making for n intervals.
    
    This runs in quadratic time: O(n^2), I believe. Why? because worst-case, each interval intersects with every other
    interval, so for each interval we have to loop through the entire list. A loop is O(n).
    
    np.sort, np.random.uniform
    
    Here's a question: Why do
    """
    N = 10_000
    n = 7  # number of intervals
    count = 0
    for _ in range(N):
        interval_lst = [np.sort(np.random.uniform(0, 1, size=2)) for _ in range(n)]
        intersect_lst = interval_lst
        for interval1 in interval_lst:
            new_intersect_lst = []
            for interval2 in intersect_lst:
                if max(interval1[0], interval2[0]) <= min(interval1[1], interval2[1]):
                    new_intersect_lst.append(interval2)
                else:
                    continue
        
            intersect_lst = new_intersect_lst
        if intersect_lst:
            count += 1
    print(count / N)
    
    
def problem_3_7():
    """
    If the number of light and dark sectors are equal in number, the probability would be the same for each hand...1/2.
    
    There is something else going on that I'm not understanding. I don't understand the 'split randomly into alternating
    dark and light sectors,' for one. Does it mean that the clocks is partitioned and then the partitions are randomly
    chosen and assigned a color, light or dark alternatively?
    
    If that's the case, then the fixed angle cannot matter.
    
    How would I do this numerically? For the position of the hand on the clock face, I could simply use the angle from
    twelve (ranging from zero to 360). The fixed angle would then be doable using a modulo of 360 (i.e., 330 + 45
    implies one hand at 330 and then other at 15). The partitions would have to be done with a dictionary, I suppose:
    the angle of the boundary as key and the value being a color. The problem is mostly just one of keeping track of
    values through the management of data structures.
    
    Is the key to this problem understanding that the probabilities under question have nothing to do with the
    indendence of the positions of the hands of the clock? In the first two scenarios, the positions of the hands are
    dependent. In the last, the positions of the clock hands are independent. Nevertheless, the probabilities requested
    are UNCONDITIONAL probabilities and so dependence is irrelevant.
    
    The book says that the EVENTS of the dark outcome is independent. This can't be true.
    """
    pass


def problem_3_8():
    """
    What Travis the Traveler needs to cross is bridges 1 and 4, 2 and 5, 1 and 3 and 5, or 2 and 3 and 4.
    
    Something I'm finding is that the words used to describe the elements of probability are throwing me off. For example,
    in this exercise, 'as likely as not' means equal probability, meaning 0.5.
    
    I'm estimating a probability of 1/2.
    
    np.random.binomial
    """
    N = 10_000_000
    cross_binary = 0
    
    for _ in range(N):
        bridges = np.random.binomial(1, 0.5, 5)
        if bridges[0] and bridges[3]:  # bridges 1 and 4
            cross_binary += 1
        elif bridges[1] and bridges[4]:  # bridges 1 and 4
            cross_binary += 1
        elif bridges[0] and bridges[2] and bridges[4]:  # bridges 1 and 4
            cross_binary += 1
        elif bridges[1] and bridges[2] and bridges[3]:  # bridges 1 and 4
            cross_binary += 1
    print(cross_binary / N)
    

def problem_3_9():
    """
    The 'most likely position' refers to what? It sounds similar to 'expected position', no? But that would be wrong.
    Somewhat fascinatingly, the expected first occurrence position is around 10.598. In contrast, the position with the
    highest probability is 1.
    
    np.where, np.random.permutation, np.unique with return_count=True for value counts.
    
    """
    N = 10_000_000
    first_occurences = []
    for _ in range(N):
        deck = ['ace'] * 4 + ['king'] * 4 + ['queen'] * 4 + ['jack'] * 4 + list(range(2, 11)) * 4
        first_occurences.append(np.where(np.random.permutation(deck) == 'ace')[0][0] + 1)  # plus one because zero indexed
    print(np.mean(first_occurences))
    print(np.unique(first_occurences, return_counts=True))


def problem_3_10ab():
    """
    What is the expected number of tosses before HT shows up for the first time? What is the expected number of tosses
    before HT shows up for the first time?
    
    What is going on here? How can these probabilities be different? I'm seeing 4 for the first (TH or HT) and 6 for the
    latter (TT or HH). How can that be?
    
    What I was getting above is correct. Fabulously, not intuitive
    """
    N = 100_000
    flips_lst = []
    
    comb = 'HT'
    # comb = 'TT'
    # comb = 'TH'
    # comb = 'HH'
    for _ in range(N):
        sequence = ''
        while True:
            sequence += np.random.choice(['H', 'T'])
            if len(sequence) > 1:
                if sequence[-2:] == comb:
                    # print(sequence[-2:], comb)
                    break
        flips_lst.append(len(sequence))
    print(np.mean(flips_lst))


def problem_3_10cd():
    """
    What is the probability that HT shows up before TT? What is the probability that HT shows up before HH?
    
    Looks like 0.75 for the first, and 0.5 for the second.
    """
    N = 100_000
    flips_lst = []
    
    count = 0
    for _ in range(N):
        sequence = ''
        while True:
            sequence += np.random.choice(['H', 'T'])
            if len(sequence) > 1:
                if sequence[-2:] == 'HH':
                    break
                elif sequence[-2:] == 'HT':
                    count += 1
                    break
    print(count / N)


def main():
    # problem_3_1()
    # problem_3_2()
    # problem_3_3()
    # problem_3_4()
    # problem_3_5()
    # problem_3_6()
    # problem_3_7()
    # problem_3_8()
    # problem_3_9()
    # problem_3_10ab()
    problem_3_10cd()


if __name__ == '__main__':
    main()
