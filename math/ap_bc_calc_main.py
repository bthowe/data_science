import numpy as np


def prob_4():
    """convergence of a definite integral, where x is upper bound of integration"""
    di = lambda x: (5 * x**2) / np.sqrt(9 * x**4 + 12 * x)
    print(di(np.linspace(1, 1e5, 100)))
    print(5/3)


def prob_6():
    """limit of infinite sum equivalent to what definite integral"""
    f = lambda n: np.sum(np.sqrt(3 + ((4 * np.arange(1, n + 1) / n) * (1 / n))))
    for n in np.arange(1, 1e5):
        print(f(n))


def main():
    # prob_4()
    prob_6()


if __name__ == '__main__':
    main()
