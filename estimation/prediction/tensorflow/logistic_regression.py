import sys
import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from itertools import product
sns.set()
matplotlib.rcParams['figure.dpi'] = 144

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def data_create():
    X = pd.read_csv('data_files/housing.csv').drop('Unnamed: 0', 1)
    y = X.pop('y').apply(lambda x: 1 if x > 20 else 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)

    return X_train, X_test, y_train, y_test


def log_reg_train_batch(X_train, y_train):
    learning_rate = .01
    training_epochs = 10000

    W = tf.Variable([[0.0]] * 13, name='weight')
    b = tf.Variable([0.0], name='bias')

    x = tf.placeholder(shape=[None, 13], dtype=tf.float32, name='x')
    y_label = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='y_label')

    y_pred = tf.matmul(x, W) + b
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_label))
    # loss = -tf.reduce_mean(tf.log(tf.nn.sigmoid(y_pred) ** y_label) + tf.log((1 - tf.nn.sigmoid(y_pred)) ** (1 - y_label)))  # these loss functions are equivalent
    # loss = -tf.reduce_sum(tf.log(tf.nn.sigmoid(y_pred) ** y_label) + tf.log((1 - tf.nn.sigmoid(y_pred)) ** (1 - y_label)))  # why does summing not give the same answer?

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):
            sess.run(optimizer, feed_dict={x: X_train, y_label: y_train.reshape(-1, 1)})
            # print(sess.run(loss, feed_dict={x: X_train, y_label: y_train.reshape(-1, 1)}))
        print(sess.run(W))
        print(sess.run(b))

def log_reg_train_mini_batch(X_train, y_train):
    y_train = y_train.values

    learning_rate = .01
    batch_size = 200
    training_epochs = 20000

    W = tf.Variable([[0.0]] * 13, name='weight')
    b = tf.Variable([0.0], name='bias')

    x = tf.placeholder(shape=[None, 13], dtype=tf.float32, name='x')
    y_label = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='y_label')

    y_pred = tf.matmul(x, W) + b
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_label))
    # loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_label))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # for epoch in range(training_epochs):
        tol = .000001
        old_loss = 1
        stop = True
        while stop:
            j = np.random.choice(len(y_train), batch_size, replace=False)
            sess.run(optimizer, feed_dict={x: X_train[j, :], y_label: y_train[j].reshape(-1, 1)})

            new_loss = sess.run(loss, feed_dict={x: X_train, y_label: y_train.reshape(-1, 1)})
            if np.abs(old_loss - new_loss) < tol:
                stop = False
            else:
                old_loss = new_loss
            print(sess.run(loss, feed_dict={x: X_train, y_label: y_train.reshape(-1, 1)}))
        print(sess.run(W))
        print(sess.run(b))

def log_reg_train_mini_batch_search(X_train, y_train):
    y_train = y_train.values

    learning_rate = [.001, .01, .1]
    batch_size = [50, 100, 150, 200]
    training_epochs = 1000

    W = tf.Variable([[0.0]] * 13, name='weight')
    b = tf.Variable([0.0], name='bias')

    x = tf.placeholder(shape=[None, 13], dtype=tf.float32, name='x')
    y_label = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='y_label')

    y_pred = tf.matmul(x, W) + b
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_label))

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
    lr = LogisticRegression(fit_intercept=True, solver='newton-cg')
    lr.fit(X_train, y_train)
    print(lr.coef_)
    print(lr.intercept_)

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = data_create()
    # log_reg_train_batch(X_train, y_train)
    log_reg_train_mini_batch(X_train, y_train)
    # log_reg_train_mini_batch_search(X_train, y_train)
    sklearn_train(X_train, y_train)
    # can't get the coefficients to match up. But these depend on the solver used.