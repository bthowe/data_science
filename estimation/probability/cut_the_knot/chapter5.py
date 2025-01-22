import sys
import math
import primefac
import itertools
import numpy as np
from functools import partial
from itertools import product
import numpy.polynomial.polynomial as poly

from optimization.mab import MAB
from optimization.mab_generalized import MAB as MAB_gen


def _monte_carlo(f, N=10_000):
    count = 0
    for _ in range(N):
        count += f()
    return count / N


def _monte_carlo2(f, N=10_000):
    count = 0
    N_count = 0

    while N_count < N:
        funcy = f()
        if funcy is not None:
            count += funcy
            N_count += 1
    return count / N


def problem_5_1():
    """
    Over 100M observations I see 0.1666, which is approximately 1/6. Over 1B observations I see
    """
    
    N = 1_000_000_000
    print(np.mean(np.where(np.sum(np.random.uniform(0, 1, size=(N, 3)), axis=1) < 1, 1, 0)))


def _triangle():
    ## point1 = 0  without loss of generality
    point2, point3 = np.sort(np.random.uniform(0, 360, size=2))
    
    angle1 = (point3 - point2) / 2
    angle2 = (360 - point3) / 2
    angle3 = point2 / 2
    
    if np.all(np.where(np.array([angle1, angle2, angle3]) < 90, 1, 0)):
        return 1
    return 0


def problem_5_2():
    """
    1/4. Why the dividing by 2 in the code above?
    """
    print(_monte_carlo(_triangle, N=1_000_000))
    

def problem_5_3():
    """
    How do I check whether the center of the circle is in a triangle? Hmmm, I don't know. The three selected points
    define a triangle as each has a line drawn between it. So I supposed checking where the origin is in relation to
    each line. I'd have to translate from polar to rectangular.
    """
    # print(_monte_carlo(_triangle, N=1_000_000))
    
    ## wlg point1 is at angle 0 and the circle has radius 1.
    point2, point3 = np.sort(np.random.uniform(0, 360, size=2))
    
    ## a / h = cos(theta 2) ==> cos(a / 1) = point2 <==> a = arccos(point2)
    x2, y2 = np.cos(point2 * np.pi / 180), np.sin(point2 * np.pi / 180)
    x3, y3 = np.cos(point3 * np.pi / 180), np.sin(point3 * np.pi / 180)
    print(x2, y2)
    print(x3, y3)
    sys.exit()
    
#     todo: okay, what's next? find closest point on line? or what about vertical/horizontal distances? And then consider cases.

def main():
    # problem_5_1()
    # problem_5_2()
    problem_5_3()


if __name__ == '__main__':
    main()