import sys
import math
import numpy as np
from itertools import product
from collections import Counter
import matplotlib.pyplot as plt


def page_105():
    """
    Numpy functions to remember: np.cumsum, np.isin for checking whether two lists contain common elements
    """
    N = 10_000_000
    first_shared_break_lst = []
    for _ in range(N):
        you = np.cumsum(np.random.choice(range(1, 6), size=100, replace=True))
        sister = np.cumsum(np.random.choice(range(1, 6), size=100, replace=True))
        
        first_shared_break_lst.append(you[np.isin(you, sister)][0])
    print(np.mean(first_shared_break_lst))


def page_109():
    """
    np.isin to check whether all elements of a_lst are the same
    """
    N = 1_000_000
    
    iterations_lst = []
    for _ in range(N):
        box = list(np.random.permutation(['r', 'b', 'g', 'y']))
    
        iterations = 0
        while ~np.all(np.isin(box, box[0])):
            first_ball = box.pop()
            box.pop()  # draw second ball
            
            box = list(np.random.permutation(box + [first_ball] * 2))
            iterations += 1
        iterations_lst.append(iterations)
    
    print(np.mean(iterations_lst))
    
    
def page_113():
    """
    I'm finding that all of these are about the same to solve. Basically, I create a list of elements and in each
    iteration change some of the elements in the list and repeat until a condition is met. Do this a whole bunch, then I
    simply find the average number of iterations. The analytical solutions are probably worth doing...but would require
    much more patience. I don't have a lot of patience for patience.
    """
    N = 1_000_000

    days_lst = []
    for _ in range(N):
        draw = 'whole'
        bottle = list(np.random.permutation(['whole'] * 99 + ['half']))
        days = 1  # because the first day is always 'whole'
        
        while draw == 'whole':
            draw = bottle.pop()
            if draw == 'whole':
                bottle = list(np.random.permutation(bottle + ['half']))
            days += 1
        days_lst.append(days)
    print(np.mean(days_lst))


def page_115():
    """
    I get 50/99 ~ 0.505 (confirmed using the Monte Carlo below). Intuitively, the first two shots yield a frequency of 1/2, the last of 1/1, and the middle
    96 which coach carl didn't see would be, in expectation, also 1/2. So the probability of making the last shot should
    be 50/99.
    
    The book gives 2/3 as the answer. What explains the discrepancy? Could it be that the coach knows he missed 96 of
    Sam the shooter's shots...and I take this into account but the author does not?
    """
    N = 1_000_000
    final_shot_lst = []
    
    for _ in range(N):
        makes = [1, 0]  # first two shots are a make and a miss
        for _ in range(3, 99):
            makes.append(np.random.binomial(1, np.mean(makes), 1)[0])
        
        makes.append(1)  # 99th shot is a make
        
        final_shot_lst.append(np.random.binomial(1, np.mean(makes), 1)[0])
    
    print(np.mean(final_shot_lst))


def main():
    # page_105()
    # page_109()
    # page_113()
    page_115()


if __name__ == '__main__':
    main()
