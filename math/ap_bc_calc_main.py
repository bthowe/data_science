import sys

import numpy as np



def integral(f, a, b, precision=10_000):
    x = np.linspace(a, b, precision)

    width = x[1] - x[0]
    height = f(x)[1:]

    return np.sum(height * width)


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



def main():
    # prob_4()
    prob_6()


if __name__ == '__main__':
    main()
