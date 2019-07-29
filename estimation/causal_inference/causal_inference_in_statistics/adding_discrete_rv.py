import sys
import numpy as np
import pandas as pd
from sympy.stats import variance, Die, covariance, correlation, density

p = 1 / 36
XZ_joint = {
    (1, 1): p, (1, 2): p, (1, 3): p, (1, 4): p, (1, 5): p, (1, 6): p,
    (2, 1): p, (2, 2): p, (2, 3): p, (2, 4): p, (2, 5): p, (2, 6): p,
    (3, 1): p, (3, 2): p, (3, 3): p, (3, 4): p, (3, 5): p, (3, 6): p,
    (4, 1): p, (4, 2): p, (4, 3): p, (4, 4): p, (4, 5): p, (4, 6): p,
    (5, 1): p, (5, 2): p, (5, 3): p, (5, 4): p, (5, 5): p, (5, 6): p,
    (6, 1): p, (6, 2): p, (6, 3): p, (6, 4): p, (6, 5): p, (6, 6): p,
}
XY_joint = {
    (1, 2): p, (1, 3): p, (1, 4): p, (1, 5): p, (1, 6): p, (1, 7): p, (1, 8): 0, (1, 9): 0, (1, 10): 0, (1, 11): 0,
    (1, 12): 0,
    (2, 2): 0, (2, 3): p, (2, 4): p, (2, 5): p, (2, 6): p, (2, 7): p, (2, 8): p, (2, 9): 0, (2, 10): 0, (2, 11): 0,
    (2, 12): 0,
    (3, 2): 0, (3, 3): 0, (3, 4): p, (3, 5): p, (3, 6): p, (3, 7): p, (3, 8): p, (3, 9): p, (3, 10): 0, (3, 11): 0,
    (3, 12): 0,
    (4, 2): 0, (4, 3): 0, (4, 4): 0, (4, 5): p, (4, 6): p, (4, 7): p, (4, 8): p, (4, 9): p, (4, 10): p, (4, 11): 0,
    (4, 12): 0,
    (5, 2): 0, (5, 3): 0, (5, 4): 0, (5, 5): 0, (5, 6): p, (5, 7): p, (5, 8): p, (5, 9): p, (5, 10): p, (5, 11): p,
    (5, 12): 0,
    (6, 2): 0, (6, 3): 0, (6, 4): 0, (6, 5): 0, (6, 6): 0, (6, 7): p, (6, 8): p, (6, 9): p, (6, 10): p, (6, 11): p,
    (6, 12): p,
}


def add_rv(joint):
    """
    joint: bivariate joint distribution, P(X, Y)

    return: P(X + Y)
    """
    Z = {}
    for k, v in joint.items():
        Z_k = k[0] + k[1]
        if Z_k in Z:
            Z[Z_k] += v
        else:
            Z[Z_k] = v
    return Z

def create_marginal(joint, ind):
    """
    joint: bivariate joint distribution, P(X, Y)
    ind: index of marginal variable, either X=0 or Y=1

    return: P(X) or P(Y)
    """
    marg_dist = {}

    for val in set(list(map(lambda x: x[ind], joint.keys()))):
        marg_dist[val] = 0
        for k, v in joint.items():
            if k[ind] == val:
                marg_dist[val] += v
    return marg_dist

def create_conditional(joint, ind, value):
    """
    joint: bivariate joint distribution, P(X, Y)
    ind: index of variable, either X=0 or Y=1, conditioning on
    value: value conditioning on

    return: P(X | Y = y) or P(Y | X = x)
    """
    a = {k: v for k, v in joint.items() if k[ind] == value}
    total_prob = sum(a.values())
    return {k[np.abs(ind - 1)]: v / total_prob for k, v in a.items()}

def E(p_x):
    """
    p_x: probability distribution for random variable X

    return: mean of r.v. X over p_x
    """
    e = 0
    for k, v in p_x.items():
        e += k * v
    return e

def V(p_x):
    """
    p_x: probability distribution for random variable X

    return: variance of r.v. X over p_x
    """
    var = 0
    for k, v in p_x.items():
        var += ((k - E(p_x)) ** 2) * v
    return var

def Cov(XY_joint, X, Y):
    mult_dist = {}
    for k, v in XY_joint.items():
        dist_k = k[0] * k[1]
        if dist_k in mult_dist:
            mult_dist[dist_k] += v
        else:
            mult_dist[dist_k] = v
    return E(mult_dist) - E(X) * E(Y)

def rho(XY_joint, X, Y):
    return Cov(XY_joint, X, Y) / (np.sqrt(V(X)) * np.sqrt(V(Y)))

def pearl_1_3_8a():
    print('(a)')
    X = {val: 1 / 6 for val in range(1, 7)}
    Z = {val: 1 / 6 for val in range(1, 7)}
    Y = add_rv(XZ_joint)

    print('E[X]: {}'.format(E(X)))
    print('\n')
    print('E[Y]: {}'.format(E(Y)))
    print('\n')
    for x in X.keys():
        print('E[Y|X = {0}] = {1}'.format(x, E(create_conditional(XY_joint, 0, x))))
        print('\n')
    for y in Y.keys():
        print('E[X|Y = {0}] = {1}'.format(y, E(create_conditional(XY_joint, 1, y))))
        print('\n')
    print('Var(X): {}'.format(V(X)))
    print('\n')
    print('Var(Y): {}'.format(V(Y)))
    print('\n')
    print('Cov(X, Y): {}'.format(Cov(XY_joint, X, Y)))
    print('\n')
    print('rho(X, Y): {}'.format(rho(XY_joint, X, Y)))
    print('\n')
    print('Cov(X, Z): {}'.format(Cov(XZ_joint, X, Z)))
    print('\n')

def pearl_1_3_8b():
    print('\n\n\n\n\n(b)')
    df = pd.DataFrame(columns=['X', 'Z', 'Y'])
    df['X'] = [6, 3, 4, 6, 6, 5, 1, 3, 6, 3, 5, 4]
    df['Z'] = [3, 4, 6, 2, 4, 3, 5, 5, 5, 5, 3, 5]
    df['Y'] = df['X'] + df['Z']

    print('e[X]: {}'.format(df['X'].mean()))
    print('\n')
    print('e[Y]: {}'.format(df['Y'].mean()))
    print('\n')
    for x in range(1, 7):
        print('E[Y|X = {0}] = {1}'.format(x, df.query('X == {}'.format(x))['Y'].mean()))
        print('\n')
    for y in range(2, 13):
        print('E[X|Y = {0}] = {1}'.format(y, df.query('Y == {}'.format(y))['X'].mean()))
        print('\n')
    print('Var(X): {}'.format(df['X'].var()))
    print('\n')
    print('Var(Y): {}'.format(df['Y'].var()))
    print('\n')
    print('Cov(X, Y): {}'.format(df[['X', 'Y']].cov()))
    print('\n')
    print('rho(X, Y): {}'.format(df[['X', 'Y']].corr()))
    print('\n')
    print('Cov(X, Z): {}'.format(df[['X', 'Z']].cov()))
    print('\n')

def pearl_1_3_8c():
    print('\n\n\n\n\n(c)')
    print('E[Y|X = 3] = {0}'.format(E(create_conditional(XY_joint, 0, 3))))

def pearl_1_3_8d():
    print('\n\n\n\n\n(d)')
    print('E[X|Y = 4] = {0}'.format(E(create_conditional(XY_joint, 1, 4))))

def pearl_1_3_8e():
    print('\n\n\n\n\n(e)')
    print('E[X|Y = 4, Z = 1] = 3')
    print('The answer is different from part (d) because Y=4 implies Z=2 and X=2 or Z=1 and X=3 or Z=3 and X=1. Conditioning on Z=1 obviously implies X=3.')

def symstat():
    """
    check out https://docs.sympy.org/latest/modules/stats.html
    """
    from sympy.stats import E

    X, Z = Die('X', 6), Die('Z', 6)
    Y = X + Z


    print('E[X]: {}'.format(E(X)))
    print('\n')
    print('E[Y]: {}'.format(E(Y)))
    print('\n')
    for x in list(density(X)):
        print('E[Y|X = {0}] = {1}'.format(x, E(Y, X=x)))  # this conditioning isn't quite right
        print('\n')
    for y in list(density(Y)):
        print('E[X|Y = {0}] = {1}'.format(y, E(X, Y=y)))
        print('\n')
    print('Var(X): {}'.format(variance(X)))
    print('\n')
    print('Var(Y): {}'.format(variance(Y)))
    print('\n')
    print('Cov(X, Y): {}'.format(covariance(X, X + Y)))
    print('\n')
    print('rho(X, Y): {}'.format(correlation(X, X + Y)))
    print('\n')
    print('Cov(X, Z): {}'.format(covariance(X, Y)))
    print('\n')
#     for some reason, these numbers are mostly incorrect

def main():
    symstat()

    pearl_1_3_8a()
    # pearl_1_3_8b()
    # pearl_1_3_8c()
    # pearl_1_3_8d()
    # pearl_1_3_8e()

if __name__ == '__main__':
    main()

# if X and Y have the same distribution, X * Y is not the same as X**2. The former is the multiplication of two rv and the latter is a transformation

# if you didn't know how Y was constructed, could you still find all of the relevant values?
# I believe the answer is no because you wouldn't know how it covaries with X, you just have it's distribution.
