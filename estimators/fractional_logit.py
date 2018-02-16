import pandas as pd

import statsmodels.api as sm

data = pd.read_table('/home/liuwensui/Documents/data/csdata.txt')

Y = data.LEV_LT3

X = sm.add_constant(data[['COLLAT1', 'SIZE1', 'PROF2', 'LIQ', 'IND3A']])

mod = sm.Logit(Y, X)

res = mod.fit()

print(res.summary())


# in R...
import pyper as pr

r = pr.R(use_pandas=True)

r.r_data = data

r('data <- rbind(cbind(r_data, y = 1, wt = r_data$LEV_LT3), cbind(r_data, y = 0, wt = 1 - r_data$LEV_LT3))')

r('mod <- glm(y ~ COLLAT1 + SIZE1 + PROF2 + LIQ + IND3A, weights = wt, subset = (wt > 0), data = data, family = binomial)')

print(r('summary(mod)'))

# https://statcompute.wordpress.com/2012/12/16/fractional-logit-model-with-python/
# when to use this model?
# comparison to similar models.