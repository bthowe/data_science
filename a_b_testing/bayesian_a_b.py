import numpy as np
import matplotlib.pyplot as plt

def calc_distributions(successes_A, failures_A, successes_B, failures_B, num_samples=1000):
    """Draws num_samples from beta distributions where alpha and beta are determined by the number of successes and
    failures. This only holds for binary outcome since beta distribution is a conjugate distribution of the
    binomial distribution"""
    A_dist = np.random.beta(1 + successes_A, 1 + failures_A, shape=num_samples)
    B_dist = np.random.beta(1 + successes_B, 1 + failures_B, shape=num_samples)
    return A_dist, B_dist

def prob_A_wins(A_dist, B_dist, wiggle=0):
    """Calculates the probability a random draw from distribution A is greater than a random draw from B plus a wiggle
    constant."""
    return np.sum(A_dist > (B_dist + wiggle)) / float(num_samples)

def plot_A_B(A_dist, B_dist):
    """Plots the distribution A and B."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1,1,1)
    ax.hist(A_dist, color='r', alpha=0.3)
    ax.hist(B_dist, color='b', alpha=0.3)
    plt.show()
