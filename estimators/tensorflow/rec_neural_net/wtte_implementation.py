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
    def __init__(self, data, target, uncensored, dropout, num_hidden=100, num_layers=2):  #dropout, num_hidden=200, num_layers=3):
        self.data = data
        self.target = target
        self.uncensored = uncensored
        self.dropout = dropout
        self._num_hidden = num_hidden
        self._num_layers = num_layers

        self._time_steps = int(self.target.shape[1])

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
        # outputs, states = tf.nn.dynamic_rnn(network, self.data, dtype=tf.float32, time_major=True)

        outputs, states = tf.nn.dynamic_rnn(network, self.data, dtype=tf.float32)

        outputs_reshaped = tf.reshape(outputs, [-1, self._num_hidden * self.data.shape[1]])  # nodes in hidden layer times the number of features which is the size of the second dimension
        outputs_dense = tf.layers.dense(outputs_reshaped, 2 * self.data.shape[1])

        return tf.concat(
            [
                # tf.multiply(5., tf.nn.sigmoid(tf.slice(outputs_dense, [0, 0], [-1, self._time_steps]))),
                # tf.exp(tf.slice(outputs_dense, [0, 0], [-1, self._time_steps])),
                tf.exp(tf.slice(outputs_dense, [0, 0], [-1, self._time_steps])),
                # tf.slice(outputs_dense, [0, 0], [-1, self._time_steps]),

                # tf.nn.sigmoid(tf.slice(outputs_dense, [0, self._time_steps], [-1, self._time_steps]))
                tf.nn.tanh(tf.slice(outputs_dense, [0, self._time_steps], [-1, self._time_steps])) + 1.
                # tf.nn.softplus(tf.slice(outputs_dense, [0, self._time_steps], [-1, self._time_steps]))
            ],
            axis=1
        )

    # todo: what I think is the problem: when I use the sigmoid activation function for beta, this only gives values between 0 and 1, which is too small to give a distribution that has a mode other than the left-most value (i.e., zero tte).
    # todo: however, initially, any other activation is giving a beta that is too large.


    def _loglikelihood(self, a_b, tte, uncensored):
        location=10.0
        growth=20.0

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
        return self._loglikelihood(self.prediction, self.data, self.uncensored)

    @lazy_property
    def optimize(self):
        learning_rate = 0.001
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        self.uncensored


        return self.cost


def _get_next_batch(batch_size, time_steps, X, y, test=False):
    '''Returns batch of covariates and corresponding outcomes of size "batch_size" and length "time_steps"'''

    x_batch = np.zeros((batch_size, time_steps))
    y_batch = np.zeros((batch_size, time_steps))

    if test:
        batch_id = [0]
    else:
        batch_id = np.random.choice(range(X.shape[0]), batch_size, replace=False)

    for t in range(batch_size):
        x_batch[t, :] = X[batch_id[t], :]
        y_batch[t, :] = y[batch_id[t], :]

    return np.expand_dims(x_batch, axis=3), np.expand_dims(y_batch, axis=3)
    # return np.expand_dims(x_batch, axis=3), y_batch


def fit_plotter(y_pred, y_true):
    l = len(y_pred)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    for i in range(l):
        ax.plot([i, i], [y_pred[i], y_true[i]], c="k", linewidth=0.5)
    ax.plot(y_pred, 'o', label='Prediction', color='cornflowerblue', alpha=.6)
    ax.plot(y_true, 'o', label='Ground Truth', color = 'firebrick', alpha=.6)
    plt.legend()
    plt.show()

def data_create(timesteps, span, num_obs):
    np.random.seed(42)

    start = np.random.randint(0, span, size=(num_obs,))

    tte_temp = []
    for _ in range(timesteps // span + 1):
        tte_temp += range(span)[::-1]
    tte_temp = np.array([tte_temp] * num_obs)

    tte = np.zeros((num_obs, timesteps))
    uncensored = np.ones((num_obs, timesteps))
    for t in range(num_obs):
        tte[t, :] = tte_temp[t, start[t]: start[t] + timesteps]
        tte[t, timesteps - start[t]:] = range(start[t], 0, -1)  #[::-1]
        uncensored[t, timesteps - start[t]:] = [0] * start[t]

    return tte, uncensored, np.where(tte == (span - 1), 1, 0)

def main():
    # data construction
    n_steps = 100
    event_span = 25
    n_obs = 1000
    data_tte, data_uncensored, data_event = data_create(n_steps, event_span, n_obs)

    # nn construction
    n_inputs = 1
    n_epochs = 1
    batch_size = 50

    data = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name='data')
    outcome = tf.placeholder(tf.float32, [None, n_steps], name='data')
    uncensored = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    dropout = tf.placeholder(tf.float32, name='dropout')
    model = SequenceClassification(data, outcome, uncensored, dropout)

    # X_test, y_test = _get_next_batch(1, n_steps, data_tte, data_uncensored, test=True)  # prep test data so don't have to do it every iteration

    # execution
    sess = tf.Session()
    tf.set_random_seed(1)
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        print('epoch {}'.format(epoch))
        for iteration in range(len(data_tte) // batch_size):
            X_batch, y_batch = _get_next_batch(batch_size, n_steps, data_tte, data_uncensored)
            # print(sess.run(model.prediction, {data: X_batch, uncensored: y_batch, dropout: 0.5}))
            # print(sess.run(model.cost, {data: X_batch, uncensored: y_batch, dropout: 0.5}))
            sess.run(model.optimize, {data: X_batch, uncensored: y_batch, dropout: 0.5})
            # print(sess.run(model.prediction, {data: X_batch, uncensored: y_batch, dropout: 0.5}))
            print(sess.run(model.cost, {data: X_batch, uncensored: y_batch, dropout: 0.5}))

        # if epoch == 0:
        #     sys.exit()

        # train_acc = sess.run(model.error, {data: X_batch, uncensored: y_batch, dropout: 1})



    X_test, y_test = _get_next_batch(1, n_steps, data_tte, data_uncensored)
    preds = sess.run(model.prediction, {data: X_test, uncensored: y_test, dropout: 1})
    scorer(preds, n_steps)


def scorer(preds, time_steps):
    import functools
    n = 10

    a = preds[0][:time_steps]
    b = preds[0][time_steps:]


    probs = np.zeros((time_steps, n))
    for i, params in enumerate(zip(a, b)):
        for j in range(n - 1, -1, -1):
            probs[i, j] = np.exp(-(j / params[0]) ** params[1]) - np.exp(-((j + 1) / params[0]) ** params[1])
            print(i, j, probs[i, j], params)
    df = pd.DataFrame(probs, columns=list(map(str, range(n - 1, -1, -1))))
    print(df)


    # n = 5
    # a = np.repeat(np.array([preds[0][:time_steps]]).T, n, axis=1)
    # b = np.repeat(np.array([preds[0][time_steps:]]).T, n, axis=1)
    #
    # tte = np.repeat(np.array([np.arange(n, 0, -1)]), time_steps, axis=0)
    #
    # df = pd.DataFrame(np.exp(-((tte / a) ** b)) - np.exp(-(((tte + 1) / a) ** b)))
    # print(df)
# todo: this is fine


if __name__ == '__main__':
    main()
    # full_data_plotter(20)


# todo: I think the problem is that my betas are not large enough.


# todo: what would the test data be, then? That is, what would the holdout data look like?
# in (neural network): covariates
# out (neural network): a, b
    # how many would be outputed?
# in (likelihood function): a, b, outcome (tte), censor variable
# out (likelihood function): likelihood score
    # how does the optimizer know what to change? It changes Variables.

# how does the data with which I'm working need to change?
    # if the event occurs after the "end" then the time period should be censored.
    # the outcome is the time to the event, not the binary event vector
    # the X is the event binary? (this makes more sense to me because we would know what happened yesterday but not the tte.

    # todo: softmax isn't working.

# what should the shape of the parameter tensor be? I'm thinking it should be 2X1
