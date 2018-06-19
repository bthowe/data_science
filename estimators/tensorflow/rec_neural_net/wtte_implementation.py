# Adapted from https://gist.github.com/danijar/c7ec9a30052127c7a1ad169eeb83f159
import sys
import random
import functools
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

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
    def __init__(self, X, y, uncensored, dropout, num_hidden=10, num_layers=1):
        self.X = X
        self.y = y
        self.uncensored = uncensored
        self.dropout = dropout
        self._num_hidden = num_hidden
        self._num_layers = num_layers

        self._time_steps = int(self.y.shape[1])

        self.prediction
        self.error
        self.optimize

    @lazy_property
    def prediction(self):
        def make_cell():
            return tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.BasicRNNCell(num_units=self._num_hidden),
                output_keep_prob=self.dropout
            )

        network = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(self._num_layers)])

        outputs, states = tf.nn.dynamic_rnn(network, self.X, dtype=tf.float32)

        outputs_reshaped = tf.reshape(outputs, [-1, self._num_hidden * self.X.shape[1]])  # nodes in hidden layer times the number of features which is the size of the second dimension

        W1 = tf.Variable(tf.zeros([self._num_hidden * self.X.shape[1], 2 * self.X.shape[1]]), name='Weights')
        b = tf.concat(
            [
                tf.multiply(tf.ones([1, self.X.shape[1]]), tf.log(4.)),
                tf.multiply(tf.ones([1, self.X.shape[1]]), 2)
            ],
            axis=1
        )
        b1 = tf.Variable(b, name='Biases')
        outputs_dense = tf.add(tf.matmul(outputs_reshaped, W1), b1)

        return tf.concat(
            [
                tf.exp(tf.slice(outputs_dense, [0, 0], [-1, self._time_steps])),
                tf.nn.softplus(tf.slice(outputs_dense, [0, self._time_steps], [-1, self._time_steps]))
            ],
            axis=1
        )

    def _loglikelihood(self, a_b, tte, uncensored):
        location = 10.0
        growth = 20.0

        loglikelihood = 0
        penalty = 0

        a = tf.slice(a_b, [0, 0], [-1, self._time_steps])
        b = tf.slice(a_b, [0, self._time_steps], [-1, self._time_steps])
        tte = tf.reshape(tte, [-1, self._time_steps])
        uncensored = tf.reshape(uncensored, [-1, self._time_steps])

        hazard0 = tf.pow(tf.div(tte + 1e-9, a), b)
        hazard1 = tf.pow(tf.div(tte + 1, a), b)

        loglikelihood += tf.reduce_mean(tf.multiply(uncensored, tf.log(tf.exp(hazard1 - hazard0) - 1.0)) - hazard1)
        penalty += tf.reduce_mean(tf.exp(tf.multiply(tf.div(growth, location), (b - location))))
        return -loglikelihood + penalty

    @lazy_property
    def cost(self):
        return self._loglikelihood(self.prediction, self.y, self.uncensored)

    @lazy_property
    def optimize(self):
        learning_rate = 0.001
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        return self.cost


def _get_next_batch(batch_size, time_steps, X, y, uncensored, test=False):
    '''Returns batch of covariates and corresponding outcomes of size "batch_size" and length "time_steps"'''

    X_batch = np.zeros((batch_size, time_steps))
    y_batch = np.zeros((batch_size, time_steps))
    uncensored_batch = np.zeros((batch_size, time_steps))

    if test:
        batch_id = [1]
    else:
        batch_id = np.random.choice(range(X.shape[0]), batch_size, replace=False)

    for t in range(batch_size):
        X_batch[t, :] = X[batch_id[t], :]
        y_batch[t, :] = y[batch_id[t], :]
        uncensored_batch[t, :] = uncensored[batch_id[t], :]

    return np.expand_dims(X_batch, axis=3), np.expand_dims(y_batch, axis=3), np.expand_dims(uncensored_batch, axis=3)


def data_create(timesteps, span, num_obs):
    start = np.random.randint(0, span, size=(num_obs,))

    tte_temp = []
    for _ in range(timesteps // span + 1):
        tte_temp += range(span)[::-1]
    tte_temp = np.array([tte_temp] * num_obs)

    tte = np.zeros((num_obs, timesteps))
    uncensored = np.ones((num_obs, timesteps))
    for t in range(num_obs):
        tte[t, :] = tte_temp[t, start[t]: start[t] + timesteps]
        tte[t, timesteps - start[t]:] = range(start[t], 0, -1)
        uncensored[t, timesteps - start[t]:] = [0] * start[t]

    return tte, uncensored, np.where(tte == (span - 1), 1, 0)


def test_plot(y_actual, preds, time_steps):
    n = 50

    a = preds[0][:time_steps]
    b = preds[0][time_steps:]

    probs = np.zeros((time_steps, n))
    for i, params in enumerate(zip(a, b)):
        for j in range(n - 1, -1, -1):
            probs[i, j] = np.exp(-(j / params[0]) ** params[1]) - np.exp(-((j + 1) / params[0]) ** params[1])
    df = pd.DataFrame(probs, columns=list(map(str, range(0, n))))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(0, len(y_actual.reshape(100,))), y_actual.reshape(100,), color='black')
    mesh = ax.imshow(df.transpose(), cmap=plt.cm.RdYlBu, origin='lower', interpolation='none', aspect='auto')
    plt.colorbar(mesh, ax=ax)
    plt.show()


def main():
    # data construction
    n_steps = 100
    event_span = 25
    n_obs = 1000
    y_array, uncensored_array, X_array = data_create(n_steps, event_span, n_obs)

    test_ind = np.random.choice(n_obs)
    print(test_ind)
    train_ind = np.arange(n_obs)
    train_ind = train_ind[train_ind != test_ind]

    y_array_test = np.expand_dims(y_array[test_ind, :], axis=0)
    uncensored_array_test = np.expand_dims(uncensored_array[test_ind, :], axis=0)
    X_array_test = np.expand_dims(X_array[test_ind, :], axis=0)
    y_array = y_array[train_ind, :]
    uncensored_array = uncensored_array[train_ind, :]
    X_array = X_array[train_ind, :]

    # nn construction
    n_inputs = 1
    n_epochs = 500
    batch_size = 50

    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name='X')
    y = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name='y')
    uncensored = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name='uncensored')
    dropout = tf.placeholder(tf.float32, name='dropout')
    model = SequenceClassification(X, y, uncensored, dropout)

    # execution
    sess = tf.Session()
    # tf.set_random_seed(1)
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        for iteration in range(len(y_array) // batch_size):
            X_batch, y_batch, uncensored_batch = _get_next_batch(batch_size, n_steps, X_array, y_array, uncensored_array)
            sess.run(model.optimize, {X: X_batch, y: y_batch, uncensored: uncensored_batch, dropout: 0.5})
        print('epoch {0}, loss {1}'.format(epoch, sess.run(model.cost, {X: X_batch, y: y_batch, uncensored: uncensored_batch, dropout: 0.5})))

    X_test, y_test, uncensored_test = _get_next_batch(1, n_steps, X_array_test, y_array_test, uncensored_array_test)
    preds = sess.run(model.prediction, {X: X_test, y: y_test, uncensored: uncensored_test, dropout: 1})
    print(y_test[0, 0, 0])
    test_plot(y_test, preds, n_steps)


if __name__ == '__main__':
    np.random.seed(44)
    main()



# in (neural network): covariates
# out (neural network): a, b
    # what is the shape of the output tensor?
        # (n_obs, time_steps * 2)
# in (likelihood function): a, b, outcome (tte), censor variable
# out (likelihood function): likelihood score
    # What does the optimizer change? The weights and biases of the neural network.


# todo next: variable input length
# todo next: try my hand at the engine data
