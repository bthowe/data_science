import numpy as np
import matplotlib.pyplot as plt

def ps22_q6():
    x = np.linspace(-5, 5, 1000000)
    y = lambda x: 0.5 * x ** 2 + 1.5 * x + 2.5

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


def ps22_q10():
    x = np.linspace(-5, 5, 1000000)
    y1 = lambda x:

    y2 = lambda x: np.exp(x)

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



def ps22_q20():
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


if __name__ == '__main__':
    ps22_q6()
    ps22_q20()