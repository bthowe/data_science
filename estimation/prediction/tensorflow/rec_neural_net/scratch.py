import sys
import sets
import joblib
import functools
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

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


class WtteRnn(object):
    # def __init__(self, data, target, dropout, num_hidden=200, num_layers=3):
    def __init__(self, data, target, mask, dropout, num_hidden=10, num_layers=1):
        self.data = data
        self.target = target
        self.mask = mask
        self.dropout = dropout
        self._num_hidden = num_hidden
        self._num_layers = num_layers

        self.prediction
        self.error
        self.optimize

    @lazy_property
    def prediction(self):
        print(self.data)
        print(self.target)
        print(self.mask)
        # sys.exit()

        masked_data = tf.boolean_mask(self.data, self.mask, axis=0)
        print(masked_data)
        print(self.data)


        network = tf.contrib.rnn.BasicLSTMCell(num_units=self._num_hidden)

        outputs, states = tf.nn.dynamic_rnn(network, self.data, dtype=tf.float32)  # tanh activation on hidenn layers?

        outputs_reshaped = tf.reshape(outputs, [-1, self._num_hidden * self.data.shape[1]])  # nodes in hidden layer times the number of features which is the size of the second dimension

        outputs_dense = tf.layers.dense(outputs_reshaped, 2)  #, activation=tf.nn.softplus)  # alpha, beta

        return tf.concat(
            [
                tf.clip_by_value(tf.exp(tf.slice(outputs_dense, [0, 0], [-1, 1])), 0, 100),
                tf.clip_by_value(tf.nn.softplus(tf.slice(outputs_dense, [0, 1], [-1, 1])), 0, 100)
            ],
            axis=1
        )


    # todo: understand the architecture of the network...what is a masking layer?


    def _loglikelihood(self, a_b, tte):
        location = 10.0
        growth = 20.0

        a_b = tf.Print(a_b, [a_b])

        a = tf.slice(a_b, [0, 0], [-1, 1])
        b = tf.slice(a_b, [0, 1], [-1, 1])
        tte = tf.cast(tf.reshape(tte, [-1, 1]), tf.float32)

        hazard0 = tf.pow(tf.div(tte + 1e-9, a), b)
        hazard1 = tf.pow(tf.div(tte + 1, a), b)
        # hazard0 = tf.Print(hazard0, [hazard0])
        # hazard1 = tf.Print(hazard1, [hazard1])

        loglikelihood = tf.reduce_mean(tf.log(tf.exp(hazard1 - hazard0) - 1.0) - hazard1)
        # penalty = tf.reduce_mean(tf.exp(tf.multiply(tf.div(growth, location), (b - location))))
        return -loglikelihood
        # return -loglikelihood + penalty

    @lazy_property
    def cost(self):
        return self._loglikelihood(self.prediction, self.target)

    @lazy_property
    def optimize(self):
        # learning_rate = 0.001
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        # return optimizer.minimize(self.cost)
        learning_rate = 0.001
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        return self.cost


def _load_file(name):
    with open(name, 'r') as file:
        return np.loadtxt(file, delimiter=',')

def _build_data(engine, time, x, max_time, is_test):
    # y[0] will be days remaining, y[1] will be event indicator, always 1 for this data
    out_y = np.empty((0, 2), dtype=np.float32)

    # A full history of sensor readings to date for each x
    out_x = np.empty((0, max_time, 24), dtype=np.float32)

    # A full history of sensor readings to date for each x
    mask = np.empty((0, max_time), dtype=np.float32)

    for i in np.sort(np.unique(engine)):
        print("Engine: " + str(int(i)))
        # When did the engine fail? (Last day + 1 for train data, irrelevant for test.)
        max_engine_time = int(np.max(time[engine == i])) + 1

        if is_test:
            start = max_engine_time - 1
        else:
            start = 0

        this_x = np.empty((0, max_time, 24), dtype=np.float32)
        this_mask = np.empty((0, max_time), dtype=np.int32)

        for j in range(start, max_engine_time):
            engine_x = x[engine == i]

            out_y = np.append(out_y, np.array((max_engine_time - j, 1), ndmin=2), axis=0)

            xtemp = np.zeros((1, max_time, 24))
            xtemp[:, max_time - min(j, 99) - 1:max_time, :] = engine_x[max(0, j - max_time + 1):j + 1, :]
            this_x = np.concatenate((this_x, xtemp))

            masktemp = np.zeros((1, max_time))
            masktemp[:, max_time - min(j, 99) - 1:max_time] = 1
            this_mask = np.concatenate((this_mask, masktemp))

        out_x = np.concatenate((out_x, this_x))
        mask = np.concatenate((mask, this_mask))

    return out_x, out_y, mask

# def _build_data(engine, time, x, max_time, is_test):
#     # y[0] will be days remaining, y[1] will be event indicator, always 1 for this data
#     out_y = np.empty((0, 2), dtype=np.float32)
#
#     # A full history of sensor readings to date for each x
#     out_x = np.empty((0, max_time, 24), dtype=np.float32)
#
#     # A full history of sensor readings to date for each x
#     mask = np.empty((0, max_time), dtype=np.float32)
#
#     for i in np.sort(np.unique(engine)):
#         print("Engine: " + str(int(i)))
#         # When did the engine fail? (Last day + 1 for train data, irrelevant for test.)
#         max_engine_time = int(np.max(time[engine == i])) + 1
#
#         if is_test:
#             start = max_engine_time - 1
#         else:
#             start = 0
#
#         this_x = np.empty((0, max_time, 24), dtype=np.float32)
#         this_mask = np.empty((0, max_time), dtype=np.int32)
#
#         for j in range(start, max_engine_time):
#             engine_x = x[engine == i]
#
#             out_y = np.append(out_y, np.array((max_engine_time - j, 1), ndmin=2), axis=0)
#
#             xtemp = np.zeros((1, max_time, 24))
#             xtemp[:, max_time - min(j, 99) - 1:max_time, :] = engine_x[max(0, j - max_time + 1):j + 1, :]
#             this_x = np.concatenate((this_x, xtemp))
#
#             masktemp = np.zeros((1, max_time))
#             masktemp[:, max_time - min(j, 99) - 1:max_time] = 1
#             this_mask = np.concatenate((this_mask, masktemp))
#
#         out_x = np.concatenate((out_x, this_x))
#         mask = np.concatenate((mask, this_mask))
#
#     return out_x, out_y, mask
#
def data_create(train_test_split_frac=.1):
    data = _load_file('../data_files/train.csv')
    # Make engine numbers and days zero-indexed
    data[:, 0:2] -= 1

    number_of_engines = int(np.max(data[:, 0])) + 1
    number_of_train_engines = int(number_of_engines * (1 - train_test_split_frac))

    train_idx = np.random.choice(number_of_engines, number_of_train_engines, replace=False)
    test_idx = [idx for idx in range(number_of_engines) if idx not in train_idx]

    train = data[np.isin(data[:, 0], train_idx)]
    test = data[np.isin(data[:, 0], test_idx)]

    # Configurable observation look-back period for each engine/day
    max_time = 100

    X_train, y_train, mask_train = _build_data(train[:, 0], train[:, 1], train[:, 2:26], max_time, False)
    X_test, y_test, mask_test = _build_data(test[:, 0], test[:, 1], test[:, 2:26], max_time, False)
    joblib.dump(X_train, '../data_files/X_train.pkl')
    joblib.dump(y_train[:, 0], '../data_files/y_train.pkl')  # I don't know why he includes the column of 1s. I believe all we need is the tte.
    joblib.dump(mask_train, '../data_files/mask_train.pkl')  # I don't know why he includes the column of 1s. I believe all we need is the tte.
    joblib.dump(X_test, '../data_files/X_test.pkl')
    joblib.dump(y_test[:, 0], '../data_files/y_test.pkl')  # I don't know why he includes the column of 1s. I believe all we need is the tte.
    joblib.dump(mask_test, '../data_files/mask_test.pkl')  # I don't know why he includes the column of 1s. I believe all we need is the tte.


def train_test_split(X, y, test_frac, normalize=True):
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    cutoff = int(X.shape[0] * test_frac)
    idx_test = idx[: cutoff]
    idx_train = idx[cutoff: ]

    if normalize:
        X_train = X[idx_train] / np.linalg.norm(X[idx_train])
        X_test = X[idx_test] / np.linalg.norm(X[idx_train])
    else:
        X_train = X[idx_train]
        X_test = X[idx_test]

    return X_train, y[idx_train], X_test, y[idx_test]

def train_test_split2(X, y, test_frac, normalize=True):
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    cutoff = int(X.shape[0] * test_frac)
    idx_test = idx[: cutoff]
    idx_train = idx[cutoff: ]

    if normalize:
        X_train = X[idx_train] / np.linalg.norm(X[idx_train])
        X_test = X[idx_test] / np.linalg.norm(X[idx_train])
    else:
        X_train = X[idx_train]
        X_test = X[idx_test]

    return X_train, y[idx_train], X_test, y[idx_test]

def _get_next_batch(batch_size, data, target):
    '''Returns batch x and y, where the y represents the same substring as x leaded by one character.'''
    idx = np.random.choice(data.shape[0], batch_size)
    return data[idx], target[idx]

def main():
    # data_create()  # X: (20631, 100, 24), y: (20631,); X_train: (18679, 100, 24), X_test: (1952, 100, 24)
    # X_train, y_train, X_test, y_test = train_test_split(joblib.load('../data_files/X.pkl'), joblib.load('../data_files/y.pkl'), .1)
    X_train = joblib.load('../data_files/X_train.pkl')
    y_train = joblib.load('../data_files/y_train.pkl')
    mask_train = joblib.load('../data_files/mask_train.pkl')
    X_test = joblib.load('../data_files/X_test.pkl')
    y_test = joblib.load('../data_files/y_test.pkl')
    mask_test = joblib.load('../data_files/mask_test.pkl')

    # todo: when making the data do I need to pad? How would I mask in tensorflow?




    X_train = X_train / np.linalg.norm(X_train)
    X_test = X_test / np.linalg.norm(X_test)

    X_train = X_train[90:91, :, :]
    y_train = y_train[90:91]
    mask_train = mask_train[90:91, :]

    sequence_length = 100
    feature_size = 24

    data = tf.placeholder(tf.float32, [None, sequence_length, feature_size])
    target = tf.placeholder(tf.int32, [None])
    mask = tf.placeholder(tf.float32, [None, sequence_length])
    dropout = tf.placeholder(tf.float32)

    model = WtteRnn(data, target, mask, dropout)


    batch_size = 50
    num_epochs = 1000

    sess = tf.Session()

    # tf.set_random_seed(1)
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epochs):

        random_idx = np.arange(X_train.shape[0])
        np.random.shuffle(random_idx)
        # for idx in np.array_split(random_idx, X_train.shape[0] // batch_size):
        #     X_batch = X_train[idx]
        #     y_batch = y_train[idx]
        #
        #     sess.run(model.optimize, feed_dict={data: X_batch, target: y_batch, dropout: .5})
        sess.run(model.optimize, feed_dict={data: X_train, target: y_train, mask: mask_train, dropout: 1})

        # train_acc = sess.run(model.error, {data: X_batch, target: y_batch, dropout: 1})
        train_acc = sess.run(model.error, {data: X_train, target: y_train, mask: mask_train, dropout: 1})
        test_acc = sess.run(model.error, {data: X_test, target: y_test, mask: mask_test, dropout: 1})
        print('Epoch {:2d} train accuracy {:3.5f}, test accuracy {:3.5f}'.format(epoch + 1, train_acc, test_acc))


if __name__ == '__main__':
    np.set_printoptions(suppress=True, threshold=10000)
    main()


# https://github.com/gm-spacagna/deep-ttf

# todo: extend this to include censoring


# really good: https://stats.stackexchange.com/questions/352036/what-should-i-do-when-my-neural-network-doesnt-learn
# gives good guidance regarding some of the implementation nuances: https://github.com/gm-spacagna/deep-ttf/

# on masking: https://www.quora.com/What-is-masking-in-a-recurrent-neural-network-RNN



# todo: start here: https://danijar.com/variable-sequence-lengths-in-tensorflow/




