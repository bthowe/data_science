import numpy as np
import matplotlib.pyplot as plt

def ps19_q8():
    x = np.linspace(-5, 5, 1000000)
    y = lambda x: x ** 3 - 3 * x ** 2 + 5 * x - 15

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y(x))

    zero_dict = {'x': 1000, 'y': 1000}
    for element in x:
        if np.abs(y(element)) < zero_dict['y']:
            zero_dict['x'] = element
            zero_dict['y'] = y(element)
    print(zero_dict)

    plt.show()



def ps19_q20():
    x = np.linspace(-5, 5, 1000000)
    y1 = lambda x: x ** 2 - 7

    y2 = lambda x: x ** 3 + x ** 2 - 5 * x - 5

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y1(x))
    ax.plot(x, y2(x))
    plt.show()

    zero_lst = []
    for element in x:
        if np.abs(y1(element) - y2(element)) < 1e-5:
            zero_lst.append((element, y1(element)))
    print(zero_lst)


def ps19_q22():
    x = np.linspace(-5, 5, 1000000)
    y = lambda x: np.sin(x)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y(x))

    if x == 0.5:
        slope = y / x
        print(slope)

    plt.show()



def ps19_q23():
    x = np.linspace(-5, 5, 1000000)
    y = lambda x: 2 ** x

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y(x))

    if x == 1:
        slope = y / x
        print(slope)

    plt.show()


if __name__ == '__main__':
    ps19_q8()
    ps19_q20()
    ps19_q22()
    ps19_q23()
    