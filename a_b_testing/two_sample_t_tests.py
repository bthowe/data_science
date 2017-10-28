import numpy as np
import scipy.stats as stats

def z_test(p1, p2, n1, n2, effect_size=0., two_tailed=True, alpha=.05):
    """Perform z-test to compare two proprtions (e.g., click through rates (ctr)).
        Note: if you set two_tailed=False, z_test assumes H_A is that the effect is
        non-negative, so the p-value is computed based on the weight in the upper tail.
        Arguments:
            p1 (float):    baseline proportion (ctr)
            p2 (float):    new proportion
            n1 (int):     number of observations in baseline sample
            n2 (int):     number of observations in new sample
            effect_size (float):    size of effect
            two_tailed (bool):  True to use two-tailed test; False to use one-sided test
                                where alternative hypothesis if that effect_size is non-negative
            alpha (float):      significance level
        Returns:
            z-score, p-value, and whether to reject the null hypothesis
    """
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    z_score = (p1 - p2) / np.sqrt(p * (1 - p) * (1. / n1 + 1. / n2))

    if two_tailed:
        p_val = (1 - stats.norm.cdf(abs(z_score))) * 2
    else:
        # Because H_A: estimated effect_size > effect_size
        p_val = 1 - stats.norm.cdf(z_score)

    reject_null = p_val < alpha
    # print 'z-score: %s, p-value: %s, reject null: %s' % (z_score, p_val, reject_null)
    return z_score, p_val, reject_null



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
