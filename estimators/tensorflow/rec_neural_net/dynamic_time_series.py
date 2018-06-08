# Adapted from https://gist.github.com/danijar/c7ec9a30052127c7a1ad169eeb83f159
import sys
import random
import functools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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
    def __init__(self, data, target, dropout, num_hidden=200, num_layers=3):
        self.data = data
        self.target = target
        self.dropout = dropout
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def prediction(self):
        # def make_cell():
        #     return tf.contrib.rnn.OutputProjectionWrapper(
        #         tf.contrib.rnn.DropoutWrapper(
        #             tf.contrib.rnn.BasicRNNCell(num_units=self._num_hidden),
        #             output_keep_prob=self.dropout
        #         ),
        #         output_size=1
        #     )
        # network = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(self._num_layers)])
        # outputs, states = tf.nn.dynamic_rnn(network, self.data, dtype=tf.float32)
        # return tf.reshape(outputs, [-1, 20])
        # # return outputs

        def make_cell():
            return tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.BasicRNNCell(num_units=self._num_hidden),
                output_keep_prob=self.dropout
            )
        network = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(self._num_layers)])
        outputs, states = tf.nn.dynamic_rnn(network, self.data, dtype=tf.float32)

        outputs_reshaped = tf.reshape(outputs, [-1, self._num_hidden * self.data.shape[1]])  # nodes in hidden layer times the n_steps which is the size of the second dimension
        return tf.layers.dense(outputs_reshaped, 20)
    #     todo: why does the latter perform so much better than the former?


    @lazy_property
    def cost(self):
        return tf.reduce_mean(tf.abs(self.prediction - self.target))


    @lazy_property
    def optimize(self):
        learning_rate = 0.001
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        return optimizer.minimize(self.cost)


    @lazy_property
    def error(self):
        return self.cost


def data_create():
    t = np.linspace(0, 6 * np.pi, 1021)
    y = np.sin(t) + np.random.normal(0, .1, size=(1021,))

    return y[: 1000].reshape([-1, 1]), y[1000:-1].reshape([-1, 1]), y[1:1001], y[1001:]


def _get_next_batch(batch_size, time_steps, X, y, test=False):
    '''Returns batch of covariates and corresponding outcomes of size "batch_size" and length "time_steps"'''

    x_batch = np.zeros((batch_size, time_steps))
    y_batch = np.zeros((batch_size, time_steps))

    if test:
        batch_id = [0]
    else:
        batch_ids = range(len(y) - time_steps - 1)
        batch_id = random.sample(batch_ids, batch_size)

    for t in range(time_steps):
        x_batch[:, t] = [X[i + t] for i in batch_id]
        y_batch[:, t] = [y[i + t] for i in batch_id]

    return np.expand_dims(x_batch, axis=3), y_batch
    # return np.expand_dims(x_batch, axis=3), np.expand_dims(y_batch, axis=3)


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
    X_train, X_test, y_train, y_test = data_create()

    # construction
    n_steps = 20
    n_inputs = 1
    n_outputs = 1
    n_epochs = 100
    batch_size = 50

    data = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    target = tf.placeholder(tf.float32, [None, n_steps])
    # target = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
    dropout = tf.placeholder(tf.float32)
    model = SequenceClassification(data, target, dropout)

    X_test, y_test = _get_next_batch(1, n_steps, X_test, y_test, test=True)  # prep test data so don't have to do it every iteration

    # execution
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        for iteration in range(len(X_train) // batch_size):
            X_batch, y_batch = _get_next_batch(batch_size, n_steps, X_train, y_train)
            sess.run(model.optimize, {data: X_batch, target: y_batch, dropout: 0.5})

        train_acc = sess.run(model.error, {data: X_batch, target: y_batch, dropout: 1})
        test_acc = sess.run(model.error, {data: X_test, target: y_test, dropout: 1})
        print('Epoch {:2d} train mae {:3.5f}, test mae {:3.5f}'.format(epoch + 1, train_acc, test_acc))

    y_pred = sess.run(model.prediction, {data: X_test, dropout: 1}).flatten()
    fit_plotter(y_pred, y_test.flatten())


if __name__ == '__main__':
    main()
