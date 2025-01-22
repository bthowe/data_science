import sys
import numpy as np
import sympy as sp
from functools import partial
from scipy.optimize import fsolve
from scipy.special import factorial


def integral(f, a, b, precision=10_000):
    x = np.linspace(a, b, precision)

    width = x[1] - x[0]
    height = f(x)[1:]

    return np.sum(height * width)


def derivative(f, x, h):
    return (f(x + h) - f(x)) / h


def prob_4():
    """convergence of a definite integral, where x is upper bound of integration"""
    di = lambda x: (5 * x**2) / np.sqrt(9 * x**4 + 12 * x)
    print(di(np.linspace(1, 1e5, 100)))
    print(5/3)


def prob_6():
    """
    Limit of infinite sum equivalent to what definite integral: see
    https://math.stackexchange.com/questions/1356472/turning-infinite-sum-into-integral
    """
    f = lambda n: np.sum(np.sqrt(3 + ((4 * np.arange(1, n + 1)) / n)) * (1 / n))
    print(f(1e6))

    a = integral(lambda x: np.sqrt(3 + 4 * x), 0, 4)
    b = integral(lambda x: np.sqrt(3 + 4 * x), 3, 7)
    c = (1 / 4) * integral(lambda x: np.sqrt(x), 3, 7, 1_000_000)
    d = integral(lambda x: np.sqrt(x), 3, 7)

    print(a, b, c, d)


def prob_7():
    # at x = 1, f is given by the function y = (5/3) * x - 1
    f = lambda x: (5/3) * x - 1
    g = lambda x: x**2 + x - 1

    d = derivative(lambda x: f(g(x)), 1, 0.00000001)
    print(d)


def f_prime(x):
    g = lambda t: 54 * t**2 - 18 * t - 8
    return derivative(g, x, 0.0001)


def _integral(x):
    g = lambda t: 54 * t**2 - 18 * t - 8
    return integral(g, 3, x)


def f_prime_solution2(x):
    return derivative(_integral, x, 0.0001)


def f_double_prime_solution2(x):
    return derivative(f_prime_solution2, x, 0.0001)


def prob_13():
    """
    Solution 1: Recognize that the derivative of the integral is just the function; then the second derivate of the
    integral is just the derivate of the function. Set equal to 0 and solve.

    Solution 2: Take the derivative of the original definite integral.

    Solution 3: Refactored form of 2, just using lambda functions and the integral and derivative functions from above.
    """
    print(fsolve(f_prime, 0.5))
    print(fsolve(f_double_prime_solution2, 0.25))

    f = lambda x: integral(lambda t: 54 * t**2 - 18 * t - 8, 3, x)
    df_dx = lambda x: derivative(f, x, 0.0001)
    d2f_dxdx = lambda x: derivative(df_dx, x, 0.0001)
    print(fsolve(d2f_dxdx, 0.25))


def _prob_18_function_wrapper(option, a):
    if option == 'A':
        f_a = partial(lambda x, n: (x ** (3 * n + 3)) / (2 ** n), a)
        return (1 / 2) * np.sum(f_a(np.arange(0, 25)))
    elif option == 'B':
        f_b = partial(lambda x, n: (x ** (3 * n + 3)) / (2 ** n), a)
        return np.sum(f_b(np.arange(0, 25)))
    elif option == 'C':
        f_c = partial(lambda x, n: (x ** (3 * n)) / (2 ** n), a)
        return 0.5 * np.sum(f_c(np.arange(0, 25)))
    else:
        f_d = partial(lambda x, n: (x ** (3 * n)) / (4 ** n), a)
        return 0.5 * np.sum(f_d(np.arange(0, 25)))


def prob_18():
    """
    The answer is A.
    """
    # a = 1
    # a = 0.5
    a = 0.25
    for option in ['A', 'B', 'C', 'D']:
        same = np.isclose(
            _prob_18_function_wrapper(option, a),
            (a ** 3) / (2 - a ** 3),
            atol=1e-6
        )
        print(option, same)


def prob_23():
    x = sp.symbols('x')

    f = (sp.exp(-3 * x) + sp.exp(3 * x)) / (6 * x)
    expr = f.series(x, 0, 5).removeO()  # removing the big O was the key and took a while to figure out that this was the issue in lambdifying
    print(expr)

    f = sp.lambdify(x, expr, modules=['sympy'])
    a = np.array(1)
    print(f(1))

#     todo: how does it do this?
#   todo: do the above using sympy and lambdify


def prob_31():
    """
    For n>15, numbers are too small, or big, for the computer to handle. It seems, though that e diverges, but 1 converges.
    """
    n = np.arange(0, 16)
    f = lambda x: (-1)**n * factorial(n) * x**n / (n**n)
    # r = 1 / np.exp(1)
    # r = 1
    r = np.exp(1)
    print(f(r))
    print(np.sum(f(r)))

    ## this was a good idea, but since there was overflow for n > 15, I needed to look at the values of the sequence to get of sense of whether it was converging or otherwise.
    # for x in np.linspace(-5, 5, 1000):
    #     print(f'{x}: {f(x)}')


def main():
    # prob_4()
    # prob_6()
    # prob_7()
    # prob_13()
    # prob_18()
    # prob_23()
    prob_31()


if __name__ == '__main__':
    main()
