import sys
import numpy as np
import pandas as pd
from functools import partial
from itertools import product
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
    'upper_left': .6,
    'upper_middle': .95,
    'upper_right': .6,
    'lower_left': .8,
    'lower_middle': .9,
    'lower_right': .8
}  # conditional on the same keeper action


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


def keeper_scorer(b):
    initial_a = [1 / 6] * 6
    cons = {'type': 'eq', 'fun': lambda x: 1 - x.sum()}
    boun = [(0, 1)] * 6
    m = minimize(keeper_func, initial_a, constraints=cons, bounds=boun, args=(b))
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

        obj_val += p_a[a] * p_b[b] * (1 - val)
    return -obj_val


def kicker_scorer(a):
    initial_b = [1/6] * 6
    cons = {'type': 'eq', 'fun': lambda x: 1 - x.sum()}
    boun = [(0, 1)] * 6
    m = minimize(kicker_func, initial_b, constraints=cons, bounds=boun, args=(a))
    return m


def total_func(a_b):
    global count
    a = a_b[:6]
    b = a_b[6:]
    print('a:', a)
    print('b:', b)

    a_hat = keeper_scorer(b)['x']
    b_hat = kicker_scorer(a)['x']

    print('a_hat:', a_hat)
    print('b_hat:', b_hat)
    print('val: ', np.mean(np.abs(a - a_hat)) + np.mean(np.abs(b - b_hat)))

    count += 1
    print(count)

    return np.sqrt(np.sum((a - a_hat) ** 2)) + np.sqrt(np.sum((b - b_hat) ** 2))


def solver():
    initial_a_b = [.2, .3, .4, .05, .05, 0, .4, .25, .15, .1, .1, 0]
    # initial_a_b = [1 / 6] * 12
    cons = [
        {'type': 'eq', 'fun': lambda x: 1 - x[:6].sum()},
        {'type': 'eq', 'fun': lambda x: 1 - x[6:].sum()}
    ]
    boun = [(0, 1)] * 12
    m = minimize(total_func, initial_a_b, constraints=cons, bounds=boun)
    # m = minimize(total_func, initial_a_b, method='L-BFGS-B', constraints=cons, bounds=boun)
    print(m)


def total_func2(a_b):
    a = a_b[:6]
    b = a_b[6:]

    a_hat = keeper_scorer(b)['x']
    b_hat = kicker_scorer(a)['x']

    df = pd.DataFrame(columns=['input', 'output'])
    df['input'] = a_b
    df['output'] = np.r_[a_hat, b_hat]
    print(df)
    return np.r_[a_hat, b_hat]


def solver2():
    from scipy.optimize import fixed_point

    initial_a_b = [.2, .3, .4, .05, .05, 0, .4, .25, .15, .1, .1, 0]
    # initial_a_b = [1 / 6] * 12

    out = fixed_point(total_func2, initial_a_b)
    print(out)


if __name__ == '__main__':
    # keeper_scorer()
    # kicker_scorer()
    solver2()

    # todo: how would we get the nash equilibrium from this?
    #   - I need to somehow find values of a and b such that each is a best response
    #   - I have each's best response function...how do I coordinate these?

    # try:
    # pass in both a and b into the solver
    # conditional on a, find best response; given this, find best response...this needs to equal a; and given a, best response needs to equal b.

    # todo: this is related to fixed points
