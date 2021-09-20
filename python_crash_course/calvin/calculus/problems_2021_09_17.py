import numpy as np
import matplotlib.pyplot as plt


def ps6_q12():
    x = np.sqrt(2)
    y = lambda x: np.sqrt(x - 1) / 1

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y(x))
    plt.show()





if __name__ == '__main__':
    ps6_q12()