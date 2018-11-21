import numpy as np
"""Algebra 1/2, Appendix E, page 419"""

def newtons_method(guess, number):
    return ((guess ** 2) + number) / (2 * guess)

guess = 4
n_iter = 4
for _ in range(n_iter):
    guess = newtons_method(guess, 15)

print('real answer: {}'.format(np.sqrt(15)))
print("newton's method: {}".format(guess))



def higher_order(guess, number, order):
    a = number / (guess ** (order - 1))
    return ((order - 1) * guess + a) / order

guess = 2
for _ in range(n_iter):
    guess = higher_order(guess, 50, 5)

print('real answer: {}'.format(50 ** (1/5)))
print("newton's method: {}".format(guess))



