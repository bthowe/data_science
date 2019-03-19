import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt

def cuts():
    return uniform.rvs(0, 360, size=6) * (np.pi / 180)


def score_intersections(names, values):
    sort_ind = np.argsort(values)
    names_sorted = np.array(names)[sort_ind]

    intersections = {'1, 5': 0, '2, 4': 0, '3': 0}
    for letter in ['a', 'b', 'c']:
        points = np.where(names_sorted == letter)[0]
        spread = points[1] - points[0]
        if spread in [1, 5]:
            intersections['1, 5'] += 1
        elif spread in [2, 4]:
            intersections['2, 4'] += 1
        else:
            intersections['3'] += 1

    return intersections

# todo: what if three parallel lines? I wouldn't give a four.


def pieces_count(intersections):
    if (intersections['3'] == 3):
        return 7
    elif (intersections['3'] == 1) and (intersections['2, 4'] == 2):
        return 6
    elif (intersections['2, 4'] == 2) and (intersections['1, 5'] == 1):
        return 5
    elif (intersections['1, 5'] == 3) or ((intersections['3'] == 1) and (intersections['1, 5'] == 2)):
        return 4
    else:
        print('scream')


def validate_pieces(value_cut_points):
    circle = plt.Circle((0, 0), 1, color='grey')
    fig, ax = plt.subplots()
    ax.add_artist(circle)
    ax.set_xlim((-2, 2))
    ax.set_ylim((-2, 2))
    ax.set_aspect('equal', adjustable='datalim')
    # plt.show()

    for pair in range(0, 6, 2):
        x1 = np.sin(value_cut_points[pair])
        y1 = np.cos(value_cut_points[pair])

        x2 = np.sin(value_cut_points[pair + 1])
        y2 = np.cos(value_cut_points[pair + 1])

        # print(value_cut_points[pair])
        # print(x1, y1)
        #
        # print(value_cut_points[pair + 1])
        # print(x2, y2)

        ax.plot((x1, x2), (y1, y2))
    plt.show()


def find_distrubtion():
    n = 1000000
    nominal_cut_points = ['a'] * 2 + ['b'] * 2 + ['c'] * 2

    history = []
    for _ in range(n):
        value_cut_points = cuts()
        intersections = score_intersections(nominal_cut_points, value_cut_points)
        history.append(pieces_count(intersections))

    print(np.mean(history))


def test():
    nominal_cut_points = ['a'] * 2 + ['b'] * 2 + ['c'] * 2
    value_cut_points = cuts()
    intersections = score_intersections(nominal_cut_points, value_cut_points)
    print(pieces_count(intersections))
    validate_pieces(value_cut_points)


def main():
    # test()
    find_distrubtion()

if __name__ == '__main__':
    main()