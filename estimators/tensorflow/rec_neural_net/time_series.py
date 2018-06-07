import sys
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


def data_create():
    t = np.linspace(0, 6 * np.pi, 1021)
    y = np.sin(t) + np.random.normal(0, .1, size=(1021,))

    return y[: 1000].reshape([-1, 1]), y[1000:-1].reshape([-1, 1]), y[1:1001], y[1001:]


def _get_next_batch(batch_size, time_steps, X, y):
    '''Returns batch of covariates and corresponding outcomes of size "batch_size" and length "time_steps"'''

    x_batch = np.zeros((batch_size, time_steps))
    y_batch = np.zeros((batch_size, time_steps))

    batch_ids = range(len(y) - time_steps - 1)
    batch_id = random.sample(batch_ids, batch_size)

    for t in range(time_steps):
        x_batch[:, t] = [X[i + t] for i in batch_id]
        y_batch[:, t] = [y[i + t + 1] for i in batch_id]

    return np.expand_dims(x_batch, axis=3), np.expand_dims(y_batch, axis=3)


def _get_next_batch_test(time_steps, X, y):
    '''Returns batch of covariates and corresponding outcomes of size "batch_size" and length "time_steps"'''

    batch_size = int(len(y) / time_steps)
    X = X.reshape(batch_size, time_steps)
    y = y.reshape(batch_size, time_steps)

    return np.expand_dims(X, axis=3), np.expand_dims(y, axis=3)


def main():
    X_train, X_test, y_train, y_test = data_create()

    # construction
    n_steps = 20
    n_inputs = 1
    n_neurons = 100
    n_outputs = 1

    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

    cell = tf.contrib.rnn.OutputProjectionWrapper(
        tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu),
        output_size=n_outputs
    )
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    learning_rate = 0.001

    loss = tf.reduce_mean(tf.abs(outputs - y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    # execution
    n_iterations = 10000
    batch_size = 50
    time_steps = 20

    with tf.Session() as sess:
        init.run()
        for iteration in range(n_iterations):
            X_batch, y_batch = _get_next_batch(batch_size, time_steps, X_train, y_train)

            # print(X_batch)
            # print(X_batch.shape)
            # sys.exit()

            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

            if iteration % 100 == 0:
                mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
                print(iteration, '\tMAE:', mse)

        X_new, y_new = _get_next_batch_test(time_steps, X_test, y_test)
        y_pred = sess.run(outputs, feed_dict={X: X_new}).flatten()

        l = len(y_pred)
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(range(l), y_pred[:l], color='cornflowerblue')
        ax.scatter(range(l), y_new.flatten()[:l], color='firebrick')
        plt.show()

if __name__ == '__main__':
    print(main())


