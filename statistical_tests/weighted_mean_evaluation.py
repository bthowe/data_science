import numpy as np
import pandas as pd


def weighted_mean(weights, values):
    return np.sum(weights * values) / np.sum(weights)

def effective_sample_size(weights):
    return np.sum(weights) ** 2 / np.sum(weights ** 2)

def variance_of_weighted_mean(weights, values):
    """use the usual unweighted estimate of the variance divided by the effective sample size. See
    http://www.analyticalgroup.com/download/Alternative%20Approaches.pdf."""
    return np.var(values) / effective_sample_size(weights)

def t_test_difference_in_weighted_means_unpooled_variances(weights1, values1, weights2, values2):
    """Calculates the t-statistic for a differences in two-population weighted means test using separate (unpooled)
    variances."""
    return (weighted_mean(weights1, values1) - weighted_mean(weights2, values2)) / \
    np.sqrt(variance_of_weighted_mean(weights1, values1) + variance_of_weighted_mean(weights2, values2))
