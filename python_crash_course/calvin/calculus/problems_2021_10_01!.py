import numpy as np
import matplotlib.pyplot as plt

def ps18_q18():
    x = np.linspace(-5, 5, 1000000)
    y = lambda x: np.sin(x) / x

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


def ps19_q6():
    x = np.linspace(-5, 5, 1000000)
    y = lambda x: ((2 + x) ** 2 - 2 ** 2) / x

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


if __name__ == '__main__':
    ps18_q18()
    ps19_q6()