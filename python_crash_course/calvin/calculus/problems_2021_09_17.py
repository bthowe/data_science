import numpy as np
import matplotlib.pyplot as plt




def ps7_q4():
    x = np.linspace(-5, 5, 1000000)
    y1 = lambda x: 0 * x

    y2 = lambda x: x ** 4 - 2 * x ** 3 + x ** 2 - x -1

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

def ps6_q12():
    x = np.sqrt(2)
    y = lambda x: np.sqrt(x - 1) / 1

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y(x))
    plt.show()





if __name__ == '__main__':
    ps6_q12()