import numpy as np

def d1(f, x):
    fd = lambda d: ((f(x + d) - f(x - d)) / (2 * d))

    d = 1
    while np.abs(fd(d) - fd(d / 2)) > epsilon:
        d *= .5

    return fd(d)

if __name__ == '__main__':
    epsilon = 0.0005
    f = lambda x: x ** 2

    print(d1(f, -4))
