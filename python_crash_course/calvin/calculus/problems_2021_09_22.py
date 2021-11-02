import numpy as np
import matplotlib.pyplot as plt

def t1_q12():
    x = np.linspace(-5, 5, 1000000)
    y1 = lambda x: - 2 * x ** 2 + 6 * x - 3


    y2 = lambda x: - x ** 4 + 4 * x ** 2 - 4

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
    t1_q12()
