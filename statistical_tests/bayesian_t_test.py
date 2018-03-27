import numpy as np
import pymc3 as pm
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
print('running on pymc3 v{}'.format(pm.__version__))

pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', false)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# from https://docs.pymc.io/notebooks/BEST.html

drug = (101,100,102,104,102,97,105,105,98,101,100,123,105,103,100,95,102,106,
        109,102,82,102,100,102,102,101,102,102,103,103,97,97,103,101,97,104,
        96,103,124,101,101,100,101,101,104,100,101)
placebo = (99,101,100,101,102,100,97,101,104,101,102,102,100,105,88,101,100,
           104,100,100,100,101,102,103,97,101,101,100,101,99,101,100,100,
           101,100,99,101,100,102,99,100,99)

y1 = np.array(drug)
y2 = np.array(placebo)
y = pd.dataframe(dict(value=np.r_[y1, y2], group=np.r_[['drug']*len(drug), ['placebo']*len(placebo)]))

y.hist('value', by='group')



μ_m = y.value.mean()
μ_s = y.value.std() * 2
σ_low = 1
σ_high = 10

with pm.model() as model:
    group1_mean = pm.normal('group1_mean', μ_m, sd=μ_s)
    group2_mean = pm.normal('group2_mean', μ_m, sd=μ_s)

    group1_std = pm.uniform('group1_std', lower=σ_low, upper=σ_high)
    group2_std = pm.uniform('group2_std', lower=σ_low, upper=σ_high)
    λ1 = group1_std**-2
    λ2 = group2_std**-2

    ν = pm.exponential('ν_minus_one', 1/29.) + 1

    group1 = pm.studentt('drug', nu=ν, mu=group1_mean, lam=λ1, observed=y1)
    group2 = pm.studentt('placebo', nu=ν, mu=group2_mean, lam=λ2, observed=y2)

    diff_of_means = pm.deterministic('difference of means', group1_mean - group2_mean)
    diff_of_stds = pm.deterministic('difference of stds', group1_std - group2_std)
    effect_size = pm.deterministic('effect size', diff_of_means / np.sqrt((group1_std**2 + group2_std**2) / 2))

    trace = pm.sample(2000)

# pm.plot_posterior(trace, varnames=['group1_mean','group2_mean', 'group1_std', 'group2_std', 'ν_minus_one'], color='#87ceeb')
pm.plot_posterior(trace, varnames=['difference of means','difference of stds', 'effect size'], ref_val=0, color='#87ceeb')
plt.show()
# pm.forestplot(trace, varnames=['group1_mean', 'group2_mean'])
# pm.forestplot(trace, varnames=['group1_std', 'group2_std', 'ν_minus_one'])
print(pm.summary(trace, varnames=['difference of means', 'difference of stds', 'effect size']))

# Thus, for the difference in means, 98.6% of the posterior probability is greater than zero, which suggests the group means are credibly different. the effect size and differences in standard deviation are similarly positive.
# These estimates suggest that the “smart drug” increased both the expected scores, but also the variability in scores across the sample. so, this does not rule out the possibility that some recipients may be adversely affected by the drug at the same time others benefit.



