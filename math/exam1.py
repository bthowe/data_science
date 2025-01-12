import sys
import numpy as np
from functools import partial


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


def prob_13():
    g = lambda t: 54 * t**2 - 18 * t - 8
    f = lambda x: integral(g, 3, x)
    # derivative(f, )
#      todo: will have to find the zero.


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


def main():
    # prob_4()
    # prob_6()
    # prob_7()
    # prob_13()  #todo
    prob_18()


if __name__ == '__main__':
    main()
