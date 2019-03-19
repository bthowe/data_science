# Adapted from https://gist.github.com/danijar/c7ec9a30052127c7a1ad169eeb83f159
import sys
import random
import functools
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import poisson

from tensorflow.python import debug as tf_debug


pd.set_option('max_columns', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper

class SequenceClassification(object):
    def __init__(self, data, target, dropout, num_hidden=100, num_layers=3):
        self.data = data
        self.target = target
        self.dropout = dropout
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.prediction
        self.optimize
        self.error

    @lazy_property
    def prediction(self):
        y = tf.layers.dense(self.data, 1, activation=None, use_bias=True, name='dense')
        with tf.variable_scope('dense', reuse=True):
            self.W = tf.get_variable('kernel')
            self.b = tf.get_variable('bias')
        return y

    # @lazy_property
    # def prediction(self):
    #     drop = tf.layers.dropout(self.data, self.dropout)
    #     deep = tf.layers.dense(drop, self.num_hidden, activation=tf.nn.relu, use_bias=True)
    #
    #     for i in range(1, self.num_layers):
    #         drop = tf.layers.dropout(deep, self.dropout)
    #         deep = tf.layers.dense(drop, self.num_hidden, activation=tf.nn.relu, use_bias=True)
    #
    #     drop = tf.layers.dropout(deep, self.dropout)
    #     return tf.layers.dense(drop, 1, activation=tf.nn.relu, use_bias=True)
    #
    # @lazy_property
    # def prediction(self):
    #     self.W = tf.Variable(tf.zeros([3, 1]), name='Weights')
    #     self.b = tf.Variable(tf.ones([1, 1]), name='Bias')
    #     return tf.add(tf.matmul(self.data, self.W), self.b)

    @lazy_property
    def cost(self):
        return tf.reduce_mean(tf.nn.log_poisson_loss(self.target, self.prediction))

    @lazy_property
    def optimize(self):
        learning_rate = 0.001
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        # return self.cost
        return tf.reduce_mean(tf.abs(self.target - self.prediction))


def data_create(num_covars, num_obs):
    b0 = 2
    b1 = 3
    b2 = 4
    b3 = 5
    beta = np.array([[b0, b1, b2, b3]])
    x_sin_int = np.random.uniform(0, 1, size=(num_obs, num_covars))
    mu = np.dot(np.c_[np.ones((num_obs, 1)), x_sin_int], beta.T)

    y = np.exp(mu + np.random.normal(0, .1, size=(num_obs, 1)))
    # y = poisson.rvs(mu)

    return x_sin_int, y


def main(X_train, y_train, X_test, y_test):
    # nn construction
    n_inputs = 1
    n_epochs = 150
    batch_size = 50

    data = tf.placeholder(shape=[None, 3], dtype=tf.float32, name='x')
    target = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='y_label')
    dropout = tf.placeholder(tf.float32, name='dropout')
    model = SequenceClassification(data, target, dropout)

    # nn execution
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        for iteration in range(n_obs_train // batch_size):
            j = np.random.choice(len(y_train), batch_size, replace=False)
            sess.run(model.optimize, {data: X_train[j, :], target: y_train[j].reshape(-1, 1)})
        train_mae = sess.run(model.error, {data: X_train, target: y_train, dropout: 0})
        test_mae = sess.run(model.error, {data: X_test, target: y_test, dropout: 0})
        print('Epoch {:2d} train mae {:3.5f}, test mae {:3.5f}'.format(epoch + 1, train_mae, test_mae))
    print(sess.run(model.b))
    print(sess.run(model.W))


def sm_poisson(X_train, y_train, X_test, y_test):
    gamma_model = sm.GLM(y_train, np.c_[np.ones((n_obs_train, 1)), X_train], family=sm.families.Poisson())
    gamma_results = gamma_model.fit()
    print(gamma_results.summary())
    print('Poisson GLM MAE: {}'.format(np.mean(np.abs(y_test - gamma_results.predict(np.c_[np.ones((n_obs_test, 1)), X_test])))))

if __name__ == '__main__':
    np.random.seed(45)

    # data construction
    n_covars = 3
    n_obs_train = 10000
    n_obs_test = 100
    X_train, y_train = data_create(n_covars, n_obs_train)
    X_test, y_test = data_create(n_covars, n_obs_test)


    main(X_train, y_train, X_test, y_test)  # MAE: 22166.60742
    sm_poisson(X_train, y_train, X_test, y_test)  # MAE: 37675.6

# if getting inf or nan
# 1. using tf.reduce_mean instead of reduce_sum?
# 2. what optimizer am I using?
# 3. are the initial values for the weights and bises in the ballpark?
# 4. change the learning rate
