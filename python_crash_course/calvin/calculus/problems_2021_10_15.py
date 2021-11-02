import matplotlib.pyplot as plt
import numpy as np



def ps28_q9():
    x = np.linspace(-5, 5, 1000000)
    y = lambda x: np.sin(np.pi / 2)
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


def ps28_q10():
    x = np.linspace(-5, 5, 1000000)
    y = np.linspace(-5, 5, 1000000)
    z = lambda x: 4 * y ** 2 - 9 * x ** 2 - 8 * y - 32

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y, z(x, y, z))

    zero_dict = {'x': 1000, 'y': 1000, 'z': 1000}
    for element in x:
        if np.abs(z(element)) < zero_dict['z']:
            zero_dict['x'] = element
            zero_dict['y'] = element
            zero_dict['z'] = z(element)
    print(zero_dict)

    plt.show()



if __name__ == '__main__':
    ps28_q10()