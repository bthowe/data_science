### Objective: draw from the conditional distribution of Y^{mis}

1. Draw from the posterior distribution of the parameters (of this conditional distribution) given the observed data, Y^{obs} and W.
    1. estimate the posterior distribution of these parameters by estimating a model
        1. estimate the posterior distributions of this model parameters
        2. use these distributions to estimate the posterior predictid distribution, which are the parameters of the conditonal distribution
2. Draw from the conditional distribution of Y^{mis}
    1. For each observation, draw from the posterior distribution of the distribution parameters found in step 1.
    2. Draw from the conditional distribution, conditional on these parameter values.
