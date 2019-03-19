import sys
import numpy as np

def distance(r, p1, p2):
    """distance calculation using latitude and longitude"""
    sig = np.arccos(np.sin(p1[0]) * np.sin(p2[0]) + np.cos(p1[0]) * np.cos(p2[0]) * np.cos(np.abs(p1[1] - p2[1])))
    return r * sig

def to_n_vec(lat_long):
    """convert point given in terms of lat. and long. to a vector"""
    return [
        np.cos(lat_long[0]) * np.cos(lat_long[1]),
        np.cos(lat_long[0]) * np.sin(lat_long[1]),
        np.sin(lat_long[0])
    ]

def distance_vec(r, a1, a2):
    """distance calculation using the positional vectors"""
    return r * np.arctan2(np.linalg.norm(np.cross(a1, a2)), np.dot(a1, a2))

def mean_vec(a1, a2):
    """calculates the midpoint of two points"""
    return (np.array(a1) + np.array(a2)) / np.linalg.norm(np.array(a1) + np.array(a2))

def trial(world_radius):
    defender_position = (np.random.uniform(-pi / 2, pi / 2), np.random.uniform(-pi, pi))  # latitude, longitude
    alien1_position = (np.random.uniform(-pi / 2, pi / 2), np.random.uniform(-pi, pi))
    alien2_position = (np.random.uniform(-pi / 2, pi / 2), np.random.uniform(-pi, pi))

    defender_v = to_n_vec(defender_position)
    alien1_v = to_n_vec(alien1_position)
    alien2_v = to_n_vec(alien2_position)

    mid_v = mean_vec(alien1_v, alien2_v)
    if (distance_vec(world_radius, alien1_v, mid_v)) < (distance_vec(world_radius, mid_v, defender_v) / 20):
        return 0  # lose
    return 1  # win

def main():
    world_radius = 4e5
    n_iter = 1000000
    outcomes = [trial(world_radius) for _ in range(n_iter)]
    print(np.mean(outcomes))

if __name__ == '__main__':
    pi = np.pi
    main()

# their answer (done analytically): 0.9927
# my answer from the 1M simulations above: 0.986938, 0.987104
# I don't know why there is a discrepancy of about 0.05.
