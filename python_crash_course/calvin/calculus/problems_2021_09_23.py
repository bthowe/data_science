import numpy as np
import matplotlib.pyplot as plt



def ps11_q6():
    x = np.linspace(-5, 5, 1000000)
    y = lambda x: - 10 * x ** 2 + 300 * x + 10000

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y(x))
    plt.show()

    zero_dict = {'x': 1000, 'y': 1000}
    for element in x:
        if np.abs(y(element)) < zero_dict['y']:
            zero_dict['x'] = element
            zero_dict['y'] = y(element)
    print(zero_dict)


def ps11_q13():
    x = np.linspace(-5, 5, 1000000)
    y1 = lambda x: 0 * x + 5
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



if __name__ == '__main__':
    ps11_q6()
    ps11_q13()