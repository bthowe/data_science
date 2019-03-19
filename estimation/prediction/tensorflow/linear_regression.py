import sys
import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from itertools import product
sns.set()
matplotlib.rcParams['figure.dpi'] = 144

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def data_create():
    X = pd.read_csv('data_files/housing.csv').drop('Unnamed: 0', 1)
    y = X.pop('y')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)

    return X_train, X_test, y_train, y_test


def ols_train_batch(X_train, y_train):
    learning_rate = .01
    training_epochs = 10000

    W = tf.Variable([[0.0]] * 13, name='weight')
    b = tf.Variable([0.0], name='bias')

    x = tf.placeholder(shape=[None, 13], dtype=tf.float32, name='x')
    y_label = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='y_label')

    y_pred = tf.matmul(x, W) + b
    loss = tf.reduce_mean(tf.square(y_pred - y_label))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):
            sess.run(optimizer, feed_dict={x: X_train, y_label: y_train.reshape(-1, 1)})
            # print(sess.run(loss, feed_dict={x: X_train, y_label: y_train.reshape(-1, 1)}))
        print(sess.run(W))
        print(sess.run(b))

def ols_train_mini_batch(X_train, y_train):
    y_train = y_train.values

    learning_rate = .01
    batch_size = 100
    training_epochs = 10000

    W = tf.Variable([[0.0]] * 13, name='weight')
    b = tf.Variable([0.0], name='bias')

    x = tf.placeholder(shape=[None, 13], dtype=tf.float32, name='x')
    y_label = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='y_label')

    y_pred = tf.matmul(x, W) + b
    loss = tf.reduce_mean(tf.square(y_pred - y_label))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):
            j = np.random.choice(len(y_train), batch_size, replace=False)
            # print(y_train[j]); sys.exit()
            sess.run(optimizer, feed_dict={x: X_train[j, :], y_label: y_train[j].reshape(-1, 1)})
            # print(sess.run(loss, feed_dict={x: X_train, y_label: y_train.reshape(-1, 1)}))
        print(sess.run(W))
        print(sess.run(b))

def ols_train_mini_batch_search(X_train, y_train):
    y_train = y_train.values

    learning_rate = [.001, .01, .1]
    batch_size = [50, 100, 150, 200]
    training_epochs = 1000

    W = tf.Variable([[0.0]] * 13, name='weight')
    b = tf.Variable([0.0], name='bias')

    x = tf.placeholder(shape=[None, 13], dtype=tf.float32, name='x')
    y_label = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='y_label')

    y_pred = tf.matmul(x, W) + b
    loss = tf.reduce_mean(tf.square(y_pred - y_label))

    eta = tf.constant(0.01, name='learning_rate')
    optimizer = tf.train.GradientDescentOptimizer(eta).minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        for e, batch in product(learning_rate, batch_size):
            sess.run(init)

            for epoch in range(training_epochs):
                j = np.random.choice(len(y_train), batch, replace=False)
                # print(y_train[j]); sys.exit()
                sess.run(optimizer, feed_dict={x: X_train[j, :], y_label: y_train[j].reshape(-1, 1), eta: e})
                # print(sess.run(loss, feed_dict={x: X_train, y_label: y_train.reshape(-1, 1)}))
            print('MSE for e={0} and batch={1}: {2}'.format(e, batch, sess.run(loss, feed_dict={x: X_train, y_label: y_train.reshape(-1, 1)})))
        # print(sess.run(W))
        # print(sess.run(b))

def sklearn_train(X_train, y_train):
    lr = LinearRegression(fit_intercept=True)
    lr.fit(X_train, y_train)
    print(lr.coef_)
    print(lr.intercept_)

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = data_create()
    # ols_train_batch(X_train, y_train)
    # ols_train_mini_batch(X_train, y_train)
    ols_train_mini_batch_search(X_train, y_train)
    sklearn_train(X_train, y_train)
