import numpy as np
import matplotlib.pyplot as plt



def ps9_q2():
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




if __name__ == '__main__':
    ps9_q2()