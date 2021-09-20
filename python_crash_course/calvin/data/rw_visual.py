import sys
import matplotlib.pyplot as plt
from random_walk import Randomwalk

while True:
    rw = Randomwalk()
    rw.fill_walk()

    plt.style.use('classic')
    fig, ax = plt.subplots(figsize=(25, 11.5))
    point_numbers = range(rw.num_points)
    ax.scatter(rw.x_values, rw.y_values, c=point_numbers, cmap=plt.cm.Blues,
               edgecolors='none', s=2)

    ax.scatter(0, 0, c='green', edgecolors='none', s=100)
    ax.scatter(rw.x_values[-1], rw.y_values[-1], c='red', edgecolors='none',
               s=100)

    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)

    plt.show()

    keep_running = input("Make another walk? (y/n): ")
    if keep_running == 'n':
        print("Remember, D & D in a week.")
        break