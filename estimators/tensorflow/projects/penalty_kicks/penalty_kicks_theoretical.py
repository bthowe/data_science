import sys
import numpy as np
import pandas as pd
from functools import partial
from itertools import product
from scipy.linalg import solve
from scipy.optimize import minimize


count = 0

p_miss = {
    'upper_left': 16 / 101,
    'upper_middle': 9 / 40,
    'upper_right': 16 / 101,
    'lower_left': 14 / 246,
    'lower_middle': 0 / 46,
    'lower_right': 14 / 246
}
p_block = {
    'upper_left': .7,
    'upper_middle': .8,
    'upper_right': .7,
    'lower_left': .7,
    'lower_middle': .85,
    'lower_right': .7
}  # conditional on the same keeper action
# nash equilibrium: [0.17548387, 0.15354839, 0.17548387, 0.17548387, 0.14451613, 0.17548387]


def keeper_func(a, b):
    actions = ['upper_left', 'upper_middle', 'upper_right', 'lower_left', 'lower_middle', 'lower_right']
    p_a = dict(zip(actions, a))
    p_b = dict(zip(actions, b))

    obj_val = 0
    for p in product(actions, actions):
        a = p[0]
        b = p[1]

        val = 0
        if a == b:
            val += p_block[b]
        val += p_miss[b]

        obj_val += p_a[a] * p_b[b] * val
    return -obj_val


def keeper_best_response(b):
    cons = {'type': 'eq', 'fun': lambda x: 1 - x.sum()}
    boun = [(0, 1)] * 6
    m = minimize(keeper_func, [1 / 6] * 6, constraints=cons, bounds=boun, args=(b))
    return m


def kicker_func(a, b):
    actions = ['upper_left', 'upper_middle', 'upper_right', 'lower_left', 'lower_middle', 'lower_right']
    p_a = dict(zip(actions, a))
    p_b = dict(zip(actions, b))

    obj_val = 0
    for p in product(actions, actions):
        a = p[0]
        b = p[1]

        val = 0
        if a == b:
            val += p_block[b]
        val += p_miss[b]

        print('a: ', a)
        print('b: ', b)
        print('val: ', val)
        print('prob: ', p_a[a] * p_b[b] * (1 - val))
        obj_val += p_a[a] * p_b[b] * (1 - val)
    return -obj_val


def kicker_best_response(a):
    cons = {'type': 'eq', 'fun': lambda x: 1 - x.sum()}
    boun = [(0, 1)] * 6
    m = minimize(kicker_func, [1 / 6] * 6, constraints=cons, bounds=boun, args=(a))
    return m


def kicker_scorer():
    a = np.array([
        [-p_block['upper_left'], p_block['upper_middle'], 0, 0, 0, 0],
        [-p_block['upper_left'], 0, p_block['upper_right'], 0, 0, 0],
        [-p_block['upper_left'], 0, 0, p_block['lower_left'], 0, 0],
        [-p_block['upper_left'], 0, 0, 0, p_block['lower_middle'], 0],
        [-p_block['upper_left'], 0, 0, 0, 0, p_block['lower_right']],
        [1, 1, 1, 1, 1, 1]
    ])

    b = np.array([0, 0, 0, 0, 0, 1])

    x = solve(a, b)
    print(x)


def keeper_scorer():
    a = np.array([
        [p_block['upper_left'], -p_block['upper_middle'], 0, 0, 0, 0],
        [p_block['upper_left'], 0, -p_block['upper_right'], 0, 0, 0],
        [p_block['upper_left'], 0, 0, -p_block['lower_left'], 0, 0],
        [p_block['upper_left'], 0, 0, 0, -p_block['lower_middle'], 0],
        [p_block['upper_left'], 0, 0, 0, 0, -p_block['lower_right']],
        [1, 1, 1, 1, 1, 1]
    ])

    b = np.array([0, 0, 0, 0, 0, 1])

    x = solve(a, b)
    print(x)


if __name__ == '__main__':
    a = [.15, .15, .15, .2, .15, .2]
    print(keeper_best_response(a))


    # keeper_best_response([1 / 6] * 6)
    # keeper_scorer()

# todo: two future questions to answer:
    # (1) can this be found by simply knowing the probabilities and not finding the FOC by using the numerical best response functions above?,
    # (2) can this be found empirically using observations?

# todo: maybe useful approach in future
# def solver2():
#     from scipy.optimize import fixed_point
#
#     initial_a_b = [1 / 6] * 12
#
#     out = fixed_point(total_func2, initial_a_b)
#     print(out)


