###Framework for causal inference 
1. there is a treatment (i.e., an intervention occurs)
2. there are multiple individuals/cases/units (i.e., in order to evaluate the potential outcomes)
3. units must not interfere with each other
4. "treatments" must be of the same type within groups
5. how are groups (controls and treatments) chosen?
6. covariates are a priori known to be unaffected by the treatment
    * unnecessary but can be beneficial
        * increase precision
        * isolate different subgroups
        * help to evaluate the assignment mechanism (evaluating balance)

Grouping these six points into broadly correlated classes, points 1 and 2 relate to the potential outcomes, 3 and 4 relate to SUTVA, 5 relates to the assignment mechanism (missing data mechanism), and 6 relates to the covariates.


### Assignment Mechanisms:
* Restrictions
    1. individualistic assignment
    2. probabilistic assignment
    3. unconfounded assignment
* Experiements versus observational studiens
    * assignment mechanism is known and controlled in the former 


### Fisher's exact p-value
Key insights: 
1. objective?
1. under the sharp null hypothesis, all the missing values can be inferred from the observed ones.
2. the test statistic is a function of the observed outcomes, the treatment assignmeent, and covariates (e.g., absolute value of the difference in average outcomes by treatment status)
3. repeat the calculations suggested in 1 and 2 for all possible assignment vectors (or for a randomly selected subset of them)
4. find the fraction of test statistics from 3 more extreme than that from 2.

### Neyman's 
Key insights:
1. ultimately interested in the population average treatment effect.
2. estimate this using the sample average treatment effect

### Difference between Fisher's and Neyman's


### Regression methods for completely randomized experiments (Chapter 7) 
The results from the least square estimator are, for the most part, only approximate, relying on large samples (i.e., large sample (consistent) but not small sample unbiased).
The conditions required for consistency follow from randomization. Advantages of regression methods include...
* incorporate covariates into estimands
* widely used and simple to implement
* yield estimates and confidence intervals of average treatment effects.
The main disadvantages is the linearity which, while not a concern in randomized experiments, can lead to sensitive estimates to minor changes in specifications in observational studies.

### Differences between the above three
(that do not have the exact (finite sample) statistical properties that make the Neyman and Fisher approaches so elegant in their simplicity but that do address more complicated questions.
assume that the randomness is not just the assignment randomness but also the sampling. 