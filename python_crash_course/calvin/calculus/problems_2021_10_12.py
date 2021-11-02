import matplotlib.pyplot as plt
import numpy as np



def ps26_q12():
    x = np.linspace(-5, 5, 1000000)
    y = lambda x: np.sqrt(x ** 2 - 2 * x - 4 * y - 4)
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
    ps26_q12()


