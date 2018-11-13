
## General Stuff

### get name of function as string
```python
function_name.__name__
```

### hasattr
```python
from sklearn.linear_model import LinearRegression
hasattr(LinearRegression, 'fit')
```
returns True. 

### 

### Nicely formated printing, such as for dictionaries
```python
import pprint
pprint.pprint(patient_info)
```

#### .format command (in what follows, the 0 is the index; it could be withheld)
align right
```python
'{0:>10}'.format('test')
```
align right, pad with underscores
```python
'{0:_>10}'.format('test')
```
align left
```python
'{0:10}'.format('test')
```
align center
```python
'{0:^10}'.format('test')
```
truncate strings
```python
'{0:.2}'.format('test')
```
truncate and pad strings
```python
'{0:4.2}'.format('test')
```
round and pad numbers
```python
'{0:5d}'.format(3141)
# yields  3141  # there is a blank in front of the number
```
```python
'{0:.3f}'.format(3.1415926535)
# yields 3.142
```
```python
'{0:6.3f}'.format(3.1415926535)
# yields  3.142  # there is a space in front of the 3
```
```python
'{0:06.3f}'.format(3.1415926535)
# yields 03.142  # now there is a zero in front of the 3
```

#### Days in a calendar month
```python
from calendar import monthrange
monthrange(2018, 6)
```

###To get a list of modules in an environment
```python
import pip
installed_packages = pip.get_installed_distributions()
installed_packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])
print(installed_packages_list)
```
### integer division
```python
divmod(13, 3)
```
returns the tuple (4, 1).
```python
13 // 3
```
returns 4. Finally, 
```python
13 % 3
```
returns the remainder, 1.

###Fix a subset of the arguments of a function
```python
from functools import partial
def func(a, b, c):
    return a + b + c
func2 = partial(func, a=2, b=3)
func2(c=3)
```
returns a value of 8

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

###Change object parameters
```python
from sklearn.ensemble import GradientBoostingRegressor
alpha = 0.05
clf = GradientBoostingRegressor(
    loss='quantile', 
    alpha=1 - alpha / 2,
    n_estimators=250, 
    max_depth=3,
    learning_rate=.1, 
    min_samples_leaf=9,
    min_samples_split=9
)
clf.set_params(alpha=alpha / 2)
```



## numpy
find substring in string in numpy array
```python
np.char.find(df['origin'].values, 'abc')
```


```python
upper = []
upper = np.concatenate((upper, rfqr.predict(X_test, quantile=98.5)))
```
#### linspace stuff
```python
np.r_[-1:2:100j]
```
is the same as 
```python
np.linspace(-1, 2, 100, endpoint=1)
```

#### add another dimension to an array
```python
np.expand_dims(x, axis=0)
```

#### repeat row
```python
np.repeat()
```

#### split array
The latter raises an exception if an equal division cannot be made.
```python
np.array_split
```
```python
np.split
```

Reshape a one dimensional array into a two dimensional array
```python
.reshape(-1, 1)
```




## pandas
mask whether values are digits in series
```python
Series.str.isdigit()
```

read the data from the clipboard into a dataframe
```python
df = pd.read_clipboard()
```


list values of a categorical variable
```python
df.value_counts()
```
convert a series to a dataframe
```python
df.to_frame()
```

.any() and .all()
```python
mask = (~(df[app_submit_cols] == '')).any(axis=1)
```


nice printing
```python
pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
```

crosstab
```python
pd.crosstab(df[0], df[1])
```

map
```python
df[0].map(lambda x: x + 1)
```
applymap
```python
df.applymap(lambda x: x + 1)
```
apply
```python
df.apply(lambda x: x[0] + 1, axis=1)
```
Map works elementwise on a series. Apply works on series or rows or columns of dataframe. Applymap applies the function elementwise to the entire dataframe.


filter observations
```python
def date_filter(x):
    if x['condition'] == 'thing':
        return True

df.loc[df.apply(date_filter, axis=1)]
```

## matplotlib
###Latex in the axis labels
```python
ax.set_ylabel('$x \cdot sin(x)$')
ax.set_xlabel('$x$')
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

###forecast library in R
```python
r = robjects.r
pandas2ri.activate()
forecast = importr('forecast')

ry_train = r.ts(df, start=r.c(1981, 1), frequency=365)

arima_fit = forecast.auto_arima(ry_train)
print(r.summary(arima_fit))
output = forecast.forecast(arima_fit, h=36)
print(output.names)
print(output.rx('lower'))
```

## scipy
### Savitzky-Golay filter
```python
from scipy.signal import savgol_filter
outcome = savgol_filter(
                x=self.X_predictions.loc[index]['predicted_target'] - self.X_predictions.loc[index].iloc[0]['predicted_target'],
                window_length=5,
                polyorder=2
            )
```

## statsmodels
### OLS
```python
import statsmodels.api as sm
mod = sm.OLS(y, X)
res = mod.fit(cov_type='HC3')
res.predict(X)  # generates dataframe of predictions

print(dir(res))
print(res.rsquared_adj)
print(res.summary())

res.get_prediction(X).summary_frame(alpha=0.05)  # confidence and prediction intervals
```
or 
```python
import patsy
import statsmodels.formula.api as smf
ols = smf.ols('foodexp ~ income', data).fit()
```
### poisson regularized regression
```python
import statsmodels.api as sm
gamma_model = sm.GLM(df[target], df[covars], family=sm.families.Poisson())
gamma_results = gamma_model.fit_regularized(alpha=.5, L1_wt=.5)
gamma_results.predict(X_test)
```
### poisson regression (without the regularization)
```python
import statsmodels.api as sm
gamma_model = sm.GLM(df[target], df[covars], family=sm.families.Poisson())
gamma_results = gamma_model.fit()
gamma_results.predict(X_test)
```
Setting alpha=0 when calling fit_regularized does not yield the same result as fit(). 

### quantile regression
```python
import patsy
import statsmodels.formula.api as smf

mod = smf.quantreg('foodexp ~ income', df)
res = mod.fit(q=.5)
print(res.summary())
```
### ARIMA
```python
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df.plot()
plot_acf(df, lags=50)
plot_pacf(df, lags=50)
plt.show()

model = ARIMA(df, order=(1, 0, 1))
model_fit = model.fit(disp=0)
print(model_fit.summary())
forecast = model_fit.forecast(steps=180, alpha=0.05)
```

### add constant
```python
X = sm.add_constant(X)
```
### get coefficients
```python
results.params
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

###Useful commands
####create a tensorflow variable
```python
x = tf.Variable(0.05, name='root', dtype=np.float64)
```
###update a variable
```python
x.assign(x ** 2)
```
####create a constant
```python
tf.constant(2, dtype=np.int64)
```
####like a constant but needs to have a value fed in each time the graph is run.
```python
tf.placeholder(np.float64, name="Inscribed_Permimeter")
```
####sum
```python
tf.add(x, y, name="sum")
```
####product
```python
tf.multiply(sum_, y, name='multiply')
```
####check whether two tensors have the same value
```python
tf.equal(tf.constant(7), tf.constant(7))
```
####cast variable as float, etc.
```python
tf.cast(tf.nn.sigmoid(y) > 0.5, np.float32)
```
####matrix multiplication
```python
tf.matmul(mat1, mat2)
```
####add across rows of a matrix
```python
tf.reduce_sum(mat, axis=0)
```
####gradient
```python
x = tf.Variable(0.05, name="root", dtype=np.float64)
f = x * x - 2
tf.gradients(f, x)[0]
```
###Sessions
####initialize
```python
sess = tf.Session()
```
####run
```python
sess.run(fibonacci)
sess.run(tf.add(x, y), feed_dict={x: 10, y: 20})
```
####close
```python
sess.close()
```

###Resets the session (closes the current section and opens a new one); this is useful in a notebook. Also resets the default graph. 
```python
def reset_tf():
    global sess
    if sess:
        sess.close()
    tf.reset_default_graph()
    sess = tf.Session()
```
###Variables must be initialized before they can be used. This is a global initializer. This function is useful in a notebook. 
```python
def reset_vars():
    sess.run(tf.global_variables_initializer())
```

#### Sample datasets
```python
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/data', one_hot=True)
```

### Layers
#### max_pooling
```python
tf.layers.max_pooling2d(out_conv, pool_size=(2, 2), strides=(2,2), padding='same')
```
#### convolutional
```python
tf.layers.conv2d(x_reshape, 16, filt_size, padding='same', activation=tf.nn.relu, name="convolution")
```
#### dense
```python
tf.layers.dense(out_pool_reshape, 100, activation=tf.nn.relu)
```
#### dropout
```python
tf.layers.dropout(hidden2, dropout, training=training)
```

#### list operations in tensorflow graph
```python
for op in graph.get_operations():
    print(op.name)
```

```python
from pylib.draw_nn import draw_neural_net_fig
draw_neural_net_fig([2, 1])
```

### activating tensorboard
After cd-ing into the correct directory.
#### in terminal
```bash
tensorboard --logdir=<directory>
```
#### in browser
```html
localhost:6006
```


#### tf debugger
```python
from tensorflow.python import debug as tf_debug
sess = tf.Session()
sess = tf_debug.LocalCLIDebugWrapperSession(sess)
```

```python
run -f has_inf_or_nan
```


## Pytorch
#### convert numpy array to tensor with type long
```python
torch.Tensor(y).long()
```
#### concatenate two column tensors (change 1 to 0 for row tensors)
```python
torch.cat((1 - output, output), 1)
```



## PyMongo
####connect to mongo
```python
from pymongo import MongoClient
client = MongoClient()
```

####create or connect to database
```python
db = client['math_book_info']
```

####delete database
```python
client.drop_database('math_book_info')
```

####create a colleciton
```python
collection = db['book_one']
```

####insert document into collection
```python
y = collection.insert_one(json_obj)
```

####print all of the documents in a collection
```python
cursor = collection.find()
for record in cursor:
    print(record)
```

