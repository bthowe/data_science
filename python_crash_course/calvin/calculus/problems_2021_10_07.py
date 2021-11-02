import numpy as np
import matplotlib.pyplot as plt

def ps23_q5():
    x = np.linspace(-5, 5, 1000000)
    y = lambda x: ((1 / 4) * x ** 2) + 1

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



def ps23_q22():
    x = np.linspace(-5, 5, 1000000)
    y = lambda x: np.exp(x ** 2) + np.sin(x) - 3

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
    ps23_q5()
    ps23_q22()