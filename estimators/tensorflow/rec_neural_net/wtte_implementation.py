# Adapted from https://gist.github.com/danijar/c7ec9a30052127c7a1ad169eeb83f159
import sys
import random
import functools
import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.python import debug as tf_debug


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
    def __init__(self, data, target, uncensored, dropout, num_hidden=200, num_layers=1):  #dropout, num_hidden=200, num_layers=3):
        self.data = data
        self.target = target
        self.uncensored = uncensored
        self.dropout = dropout
        self._num_hidden = num_hidden
        self._num_layers = num_layers
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
        outputs, states = tf.nn.dynamic_rnn(network, self.data, dtype=tf.float32)

        outputs_reshaped = tf.reshape(outputs, [-1, self._num_hidden * self.data.shape[1]])  # nodes in hidden layer times the number of features which is the size of the second dimension
        outputs_dense = tf.layers.dense(outputs_reshaped, 2)

        return tf.stack(
            [
                tf.exp(tf.reshape(tf.slice(outputs_dense, [0, 0], [-1, 1]), [-1])),
                tf.nn.softplus(tf.reshape(tf.slice(outputs_dense, [0, 1], [-1, 1]), [-1]))
                # tf.nn.elu(tf.reshape(tf.slice(outputs_dense, [0, 1], [-1, 1]), [-1]))
            ],
            axis=1
        )
        # return tf.layers.dense(outputs_reshaped, 2)  # output layer of size two because of I am predicting the two parameters in the Weibull distribution

    def _loglikelihood(self, a_b, tte, uncensored):
        loglikelihood = 0
        for i in range(50):
            a = tf.reshape(tf.slice(a_b, [i, 0], [1, 1]), [])
            b = tf.reshape(tf.slice(a_b, [i, 1], [1, 1]), [])
            tte_i = tf.reshape(tf.slice(tte, [i, 0, 0], [1, -1, -1]), [-1])
            uncensored_i = tf.reshape(tf.slice(uncensored, [i, 0], [1, -1]), [-1])

            hazard0_i = tf.pow(tf.div(tte_i + 1e-9, a), b)
            hazard1_i = tf.pow(tf.div(tte_i + 1.0, a), b)

            # todo: it must be the case that hazard1 is smaller than hazard0.  How can that be true?


            loglikelihood += tf.reduce_sum(tf.multiply(uncensored_i, tf.log(tf.exp(hazard1_i - hazard0_i) - 1.0)) - hazard1_i)

        return -loglikelihood

    @lazy_property
    def cost(self):
        return self._loglikelihood(self.prediction, self.data, self.uncensored)

    @lazy_property
    def optimize(self):
        learning_rate = 0.001
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        self.uncensored


        return self.cost


def data_create(timesteps):
    np.random.seed(42)

    span = 25
    num_obs = 1000

    start = np.random.randint(0, span, size=(num_obs,))

    tte_temp = []
    for _ in range(timesteps // span):
        tte_temp += range(span)[::-1]
    tte_temp = np.array([tte_temp] * num_obs)
    censor_temp = np.array([[1] * span * (timesteps // span) + [0] * span] * num_obs)

    tte = np.zeros((num_obs, timesteps))
    censor = np.zeros((num_obs, timesteps))
    for t in range(num_obs):
        tte[t, :] = np.concatenate((tte_temp[t, start[t]: start[t] + timesteps], np.array(range(1, start[t] + 1))[::-1]))
        censor[t, :] = censor_temp[t, start[t]: start[t] + timesteps]

    return tte, censor, np.where(tte == 0, 1, 0)
    # return np.expand_dims(tte, axis=3), censor, np.where(tte == 0, 1, 0)


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

    return np.expand_dims(x_batch, axis=3), y_batch


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


def main():
    n_steps = 100
    data_tte, data_uncensored, data_event = data_create(n_steps)

    # construction
    n_inputs = 1
    n_outputs = 2
    n_epochs = 10
    batch_size = 50

    data = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name='data')
    uncensored = tf.placeholder(tf.float32, [None, n_steps])
    a_b_params = tf.placeholder(tf.float32, [None, n_outputs])
    # data = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name='data')
    # uncensored = tf.placeholder(tf.float32, [None, n_steps])
    # a_b_params = tf.placeholder(tf.float32, [None, n_steps])
    dropout = tf.placeholder(tf.float32, name='dropout')
    model = SequenceClassification(data, a_b_params, uncensored, dropout)

    # X_test, y_test = _get_next_batch(1, n_steps, data_tte, data_uncensored, test=True)  # prep test data so don't have to do it every iteration

    # execution
    sess = tf.Session()
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        print('epoch {}'.format(epoch))
        for iteration in range(len(data_tte) // batch_size):
            X_batch, y_batch = _get_next_batch(batch_size, n_steps, data_tte, data_uncensored)
            print(sess.run(model.prediction, {data: X_batch, uncensored: y_batch, dropout: 0.5}))
            print(sess.run(model.cost, {data: X_batch, uncensored: y_batch, dropout: 0.5}))
            sess.run(model.optimize, {data: X_batch, uncensored: y_batch, dropout: 0.5})
            print(sess.run(model.prediction, {data: X_batch, uncensored: y_batch, dropout: 0.5}))
            print(sess.run(model.cost, {data: X_batch, uncensored: y_batch, dropout: 0.5}))
            sys.exit()

        # train_acc = sess.run(model.error, {data: X_batch, uncensored: y_batch, dropout: 1})

        print(sess.run(model.prediction, {data: X_batch, uncensored: y_batch, dropout: 0.5}))
    sys.exit()

        # test_acc = sess.run(model.error, {data: X_test, uncensored: y_test, dropout: 1})
        # print('Epoch {:2d} train mae {:3.5f}, test mae {:3.5f}'.format(epoch + 1, train_acc, test_acc))

    # todo: see how it is at predicting the events

    # tf.train.Saver().save(sess, './ts_model')
    #
    # y_pred = sess.run(model.prediction, {data: X_test, dropout: 1}).flatten()
    # fit_plotter(y_pred, y_test.flatten())


def forecast(periods):
    X_train, X_test, y_train, y_test = data_create()
    X = X_train[-20:, :]  # remember this 20 here and below is a hyperparameter in the model.

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('ts_model.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))

        graph = tf.get_default_graph()
        pred = graph.get_tensor_by_name('dense/BiasAdd:0')
        data = graph.get_tensor_by_name('data:0')
        dropout = graph.get_tensor_by_name('dropout:0')

        forecasts = []
        for _ in range(periods):
            X_batch, y_batch = _get_next_batch(1, 20, X, y_train[-20:], test=True)
            next_pred = sess.run(pred, {data: X_batch, dropout: 1}).flatten()[-1]
            forecasts.append(next_pred)
            X = np.r_[X[1:, :], [[next_pred]]]

    # fit_plotter(X.flatten()[-periods:], y_test.flatten()[:periods])
    # print('MAE of forecasts: {0:.3f}'.format(np.mean(np.abs(X.flatten()[-periods:] - y_test.flatten()[:periods]))))
    return forecasts


def full_data_plotter(forecast_periods):
    X_train, X_test, y_train, y_test = data_create()

    y_forecast = forecast(forecast_periods)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(len(X_train)), X_train, color='cornflowerblue')
    ax.plot(range(len(X_train), len(X_train) + len(X_test)), X_test, color='seagreen')
    ax.plot(range(len(X_train), len(X_train) + forecast_periods), y_forecast, color='firebrick')
    plt.show()

if __name__ == '__main__':
    main()
    # full_data_plotter(20)

