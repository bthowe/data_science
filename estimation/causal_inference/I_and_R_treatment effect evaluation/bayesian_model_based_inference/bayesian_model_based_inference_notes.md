### Objective finite-sample: draw from the conditional distribution of Y^{mis}
1. Draw from the posterior distribution of the parameters (of this conditional distribution) given the observed data, Y^{obs} and W.
    1. estimate the posterior distribution of these parameters by estimating a model
        1. estimate the posterior distributions of this model parameters
        2. use these distributions to estimate the posterior predictid distribution, which are the parameters of the conditonal distribution
2. Draw from the conditional distribution of Y^{mis}
    1. For each observation, draw from the posterior distribution of the distribution parameters found in step 1.
    2. Draw from the conditional distribution, conditional on these parameter values.


### Objective super-population: draw from the conditional distribution of Y^{mis}
1. Draw from the posterior distribution of the parameters given the observed data, Y^{obs} and W.
    1. estimate the posterior distribution of these parameters by estimating a model
        1. estimate the posterior distributions of this model parameters
        2. use these distributions to estimate the posterior predicted distribution, which are the parameters of the conditional distribution
2. The average and sample variance of these draws gives estimates of the posterior mean and variance of the population average treatment effect.
    1. For each draw, calculate the treatment effect
    2. Calculate the mean and variance.


### Difference in approaches between finite sample and super-population
1. Finite-sample
    1. In order to calculate the treatment effect you need to impute the missing (unobserved) values.
    2. Draw from the conditional distribution of Y^{mis} conditional on theta.
2. Super-population (when the normal linear model is used.)
    1. Imputing isn't necessary because you take the expectation. You can estimate the expectation without bias by taking the sample average because the treatment assignment is random.
    2. No need to draw from the conditional distribution of Y^{mis} conditional on theta, since the expectation reduces this to simply using theta.

On page 172, there is some discussion surrounding how estimated variances compare between these two paradigms, specifically,
the role of correlation between the two sets of outcomes. Perfect correlation increases the variance. The super-population estimate
of variance of the treatment effect is closer to the finite-sample case when the outcomes are perfectly correlated. 

### R.V. Notes
1. Is this thing a random variable?
2. Why is it a random variable?
3. How can I describe its inherent uncertainty?
    1. Sample mean
        1. Is it unbiased? consistent?
        2. These are relevant when trying to link the estimate with the true parameter
    2. Sample variance
        1. Is it unbiased? consistent?
        2. These are relevant when trying to link the estimate with the true parameter
    3. Distribution
        1. Can I use the central limit theorem? I.e., approximate normality?
        2. Assume some distribution
