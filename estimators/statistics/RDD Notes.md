Analysis of Covariance (ANCOVA)

- If the outcome variable is continuous, then an ordinary regression equation can be used.
- If the outcome is binary, then use a logistic regression
    -  P(Y_i = 1) = 1 / (1 + e^{-U_i}), where U_i = \hat{\beta}_0 + \hat{\beta}_1 * Z_i + \hat{\beta}_2 (X_i - X_c)
    - Here, 
        - Z_i is the treatment dummy variable (i.e., to the left or right of the cutoff)
        - X_i is the assignment variable (e.g., the "running" or "forcing" variable)
        - X_c is the cutoff
        - \hat{\beta}_0 is the intercept
        - \hat{\beta}_1 is the estimate of the treatment effect
        - \hat{\beta}_2 is the estimate of the effect of the assignment on the outcome
- I should explore using nonlinear function sof the assignment variable in the equation, such as higher orders or an interaction.
    - Higher order: e.g., U_i = \hat{\beta}_0 + \hat{\beta}_1 * Z_i + \hat{\beta}_2 (X_i - X_c) + \hat{\beta}_3 (X_i - X_c)^2
    - Interaction term between the treatment assignment and the assignment variable
        - This would allow for different slopes on either side of the cutoff
            U_i = \hat{\beta}_0 + \hat{\beta}_1 * Z_i + \hat{\beta}_2 (X_i - X_c) + \hat{\beta}_3 Z_i (X_i - X_c)
    - These can be combined
        -e.g., U_i = \hat{\beta}_0 + \hat{\beta}_1 * Z_i + \hat{\beta}_2 (X_i - X_c) + \hat{\beta}_3 Z_i (X_i - X_c) + \hat{\beta}_4 (X_i - X_c)^2 + \hat{\beta}_5 Z_i (X_i - X_c)^2
    - Which terms should I add?
        - If I have enough data, use cross-validation on the dataset to determine which terms to include
        - Should I make a train/test split of the data?
            - The purpose of the test data is to see how future data will act in the model.
            - But what I care about here are the coefficients of the estimated model (done using the training data), so test data is superfluous. 
        - Could use observation if relatively few data points. 
        - How sensitive are the conclusions to which terms are added?
        - When in doubt, start by overfitting the model
            - try and eliminate the nonsignificant terms.
            - when in doubt, leave terms in: results will be unbiased at the cost of a reduction in power.
            
- Reduced power in RDD relative to RCT (randomized controlled trial) because you have fewer observations
    - You don't get to use the observations to the right (or left, respectively) of the cutoff to fit the line because they are counterfactual.
    
- How to generalize beyond the local average treatment effect?


What is meant by pretest?
- I think it means literally a test administered before the natural experiment to all of the observations.





Polynomial Regression
1. Even if this polynomial model is not the real one, it might still be a good approximation for E(Y|X = x) = h(x). 
    1. From Stone-Weierstrass theorem, if h() is continuous on some interval, then there is a uniform approximation of it by polynomial functions.
2. Problems
    1. Overfit
        1. Too many "oscillations"
        2. Not robust...sensitive to changes in the values of any given point 
3. 


Non-parametric regression
1. Kernal weighted averages
    1. Nadaraya-Watson kernel estimator...fits a constant
        1. Consider as a baseline, the local average model: \hat{f}(x_0) = \sum_i (y_i * I(|x_i - x_0| < h)) / \sum_i I(|x_i - x_0| < h)
            1. This will lead to discontinuous estimates
        2. Generalize this with the Nadaraya-Watson kernel estimator
            1. \hat{f}(x_0) = \sum_i (y_i * K_h(x_i, x_0)) / \sum_i K_h(x_i, x_0)
        3. Use cross-validation to find bandwidth that minimizes the loss function (mean squared error)
            1. could use K-fold cross-validation or leave-one-out
        4. this estimator suffers from bias both at the boundaries and in the interior when the x_i's are not uniformly distributed.
            1. There are fewer data points at either extremes
2. Local linear regression...fits a straight line locally
    1. Mitigates (up to first order) the bias of kernel estimators which only 
    
3. LOWESS (locally weighted scatterplot smoothing)
4. LOESS (locally estimated scatterplot smoothing)
    1. Denotes a method that is also known as locally weighted polynomial regression. 
    2. At each point in the range of the data set a low-degree polynomial is fitted to a subset of the data, with explanatory variable values near the point whose response is being estimated. 
    3. The polynomial is fitted using weighted least squares, giving more weight to points near the point whose response is being estimated and less weight to points further away. 
    4. The value of the regression function for the point is then obtained by evaluating the local polynomial using the explanatory variable values for that data point. 
    5. The LOESS fit is complete after regression function values have been computed for each of the n data points.
    

The bandwidth, h, is a scale parameter: K_h(u) = (1/h)K(u/h).
Thus, if the Epanechnikov kernel is used, K_h(u) = (1/h)(3/4)(1-(u/h)^2), K(u) = (3/4)(1 - u^2), where the support is |u| <= 1 (thus, the function is zero-valued otherwise).






