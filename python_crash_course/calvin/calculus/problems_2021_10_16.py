import matplotlib.pyplot as plt
import numpy as np



def ps29_q7():
    x = np.linspace(-5, 5, 1000000)
    y1 = lambda x: np.sqrt(x)

    y2 = lambda x: 0.25 * x + 1

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
    ps29_q7()