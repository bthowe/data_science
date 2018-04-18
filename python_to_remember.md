
To get a list of modules in an environment
```python
import pip
installed_packages = pip.get_installed_distributions()
installed_packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])
print(installed_packages_list)
```

## sklearn
### sample datasets
```python
from sklearn import datasets
iris = datasets.load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

```
### Pipeline
Use all but the last step in a pipeline
```python
Pipeline(my_pipeline.steps[:-1]). transform()
```

## pandas
list values of a categorical variable
```python
df.value_counts()
```
convert a series to a dataframe
```python
df.to_frame()
```


## rpy2
```python
os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

r = robjects.r
pandas2ri.activate()
wru = importr('wru')

r['load']('/Users/travis.howe/Downloads/voters.RData')
print(r['voters'])

dfr = pandas2ri.py2ri(df.head())
print(dfr)

X = r['voters']
print(X.append(pd.DataFrame([['11', 'Howe', 'MO', '5', '095', '000000', '0000', '0', '35', '0', 'Ind', '0', '0000']], columns=X.columns.tolist())))
print(X[['surname', 'state', 'county', 'tract', 'block']])

print(dir(wru))
X_out = wru.predict_race(voter_file=X, census_geo='county', census_key=CENSUS_KEY, party='PID')
print(pandas2ri.ri2py(X_out))
```



## statsmodels
```python
import statsmodels.api as sm
mod = sm.OLS(y, X)
res = mod.fit(cov_type='HC3')

print(dir(res))
print(res.rsquared_adj)
print(res.summary())
```


## PYMC3
```python
import pymc3 as pm
```

#### Working with the trace 
```python
trace.varnames
```

Implements the Gelman-Rubin diagnostic test for lack of convergence.
```python
pm.diagnostics.gelman_rubin(trace)['column_name']
```

#### Deterministic
This creates a deterministic random variable, which implies that its value is completely determined by its parents’ values. 
That is, there is no uncertainty beyond that which is inherent in the parents’ values.
This can be achieved by specifying a relationship such as the following linear relationship or wrapping draws in named 
Deterministic objects.

```python

mu = alpha + beta[0] * X1 + beta[1] * X2

pm.Deterministic('difference of means', group1_mean - group2_mean)
```


####Diagnostics
https://docs.pymc.io/api/diagnostics.html

### Plotting
```python
pm.plot_posterior(trace, varnames=['group1_mean','group2_mean', 'group1_std', 'group2_std', 'ν_minus_one'], color='#87ceeb');
pm.plot_posterior(trace, varnames=['difference of means','difference of stds', 'effect size'], ref_val=0, color='#87ceeb');
```

When the following is called on a trace with more than one chain, it plots the potential scale reduction parameter, which
is used to reveal evidence for lack of convergence. Value near one suggest the model hsa converged.
```python
pm.forestplot(trace, varnames=['group1_mean', 'group2_mean']);
```

### Summary
For the variables specified, the following plots the mean, standard deviation, mc error, .025 and .975 quantiles, R hat (the Gelman-Rubin statistic), and the estimate of the effective sample size of a set of traces
```python
pm.summary(trace, varnames=['difference of means', 'difference of stds', 'effect size'])
```

### KDE
```python
pm.kdeplot(np.random.exponential(30, size=10000), shade=0.5);
```








## Tensorflow

#### Sample datasets
```python
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/data', one_hot=True)
```



```python
from pylib.draw_nn import draw_neural_net_fig
draw_neural_net_fig([2, 1])
```
