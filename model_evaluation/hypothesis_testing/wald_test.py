import sys
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import chi2


def Wald_test(theta, V, c, n):
    """
    Calculates the test-statistic and corresponding p-value for the Wald test.

    :param theta: A (0, p) numpy array of estimates of the parameters.
    :param V: The p x p variance-covariance matrix associated with the estimates in theta.
    :param c: An r-dimenional tuple whose elements are functions of theta. E.g., c = (f1, f2) or c(f1, ), where f1 is something like
    f1 = lambda x: x[0] + x[1] - 10. This function represents the nonlinear hypotheses.
    :param n: The sample size.
    :return: A dictionary with the value of the test statistic (distributed as chi2 with r degrees of freedom) and its
    corresponding p-value.
    """
    r = len(c)
    p = len(theta)


    c_theta = np.array([f(theta) for f in c]).reshape((r, 1))  # r x 1

    c_jacobian = np.empty((0, p), float)  # r x p, where r is the dimension of c and p the dimension of theta. r is the number of degrees of freedom.
    for f in c:
        eps = np.sqrt(np.finfo(float).eps)
        c_jacobian = np.append(c_jacobian, optimize.approx_fprime(theta, f, eps).reshape((1, p)), axis=0)

    test_stat = float(n * c_theta.T.dot(np.linalg.inv(c_jacobian.dot(V).dot(c_jacobian.T))).dot(c_theta))
    p_val = 1 - chi2.cdf(test_stat, 1)

    return {'test statistic': test_stat, 'p-value': p_val}

if __name__ == '__main__':
    V = [[10, 5], [5, 10]]
    theta = np.array([5, 6])
    n = 90

    # b1==0
    # f1 = lambda x: x[0]
    # c = (f1,)

    # # b1==0 and b2==0
    # f1 = lambda x: x[0]
    # f2 = lambda x: x[1]
    # c = (f1, f2)
    #
    # # b1==b2
    # f1 = lambda x: x[0] - x[1]
    # c = (f1,)
    #
    # # b1 + b2 = 10 and b1^2 + b2^2 = 10
    # f1 = lambda x: x[0] + x[1] - 10
    # f2 = lambda x: x[0] ** 2 + x[1] ** 2 - 10
    # c = (f1, f2)

    # # b1 + b2 = 10
    f1 = lambda x: x[0] + x[1] - 10
    c = (f1,)

    print(Wald_test(theta, V, c, n))