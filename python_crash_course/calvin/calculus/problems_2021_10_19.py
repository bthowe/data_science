import matplotlib.pyplot as plt
import numpy as np



def ps30_q12():
    x = np.linspace(-5, 5, 1000000)
    y = lambda x: (np.exp(2 + x) - np.exp(2)) / x

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


def ps30_q23():
    x = np.linspace(-5, 5, 1000000)
    y1 = lambda x: np.sqrt(- x ** 2 + 8)

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
    ps30_q12()
    ps30_q23()