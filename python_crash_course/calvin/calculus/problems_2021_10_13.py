import matplotlib.pyplot as plt
import numpy as np



def ps27_q14():
    x = np.linspace(-5, 5, 1000000)
    y = np.linspace(-5, 5, 1000000)
    z = lambda x, y: 4 * y ** 2 - 9 * x ** 2 - 8 * y - 32

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y, z(x, y))

    zero_dict = {'x': 1000, 'y': 1000, 'z': 1000}
    for element in x:
        if np.abs(z(element)) < zero_dict['z']:
            zero_dict['x'] = element
            zero_dict['y'] = z(element)
    print(zero_dict)

    plt.show()




if __name__ == '__main__':
    ps27_q14()