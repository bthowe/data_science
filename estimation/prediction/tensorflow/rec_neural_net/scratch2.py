import sys
import sets
import joblib
import functools
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import normalize
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
    def __init__(self, data, target, dropout, num_hidden=20, num_layers=1):
        self.data = data
        self.target = target
        self.dropout = dropout
        self._num_hidden = num_hidden
        self._num_layers = num_layers

        self.prediction
        self.error
        self.optimize


    @lazy_property
    def length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    @lazy_property
    def prediction(self):
        network = tf.contrib.rnn.BasicLSTMCell(num_units=self._num_hidden)

        outputs, _ = tf.nn.dynamic_rnn(
            network,
            self.data,
            dtype=tf.float32,
            sequence_length=self.length
        )

        # outputs = tf.transpose(outputs, [1, 0, 2])
        # last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)
        last = self._last_relevant(outputs, self.length)
        # print(last)

        # out1 = tf.contrib.layers.fully_connected(last, 10, activation_fn=tf.nn.tanh)

        out = tf.contrib.layers.fully_connected(last, 2)
        # return tf.contrib.layers.fully_connected(last, 2, activation_fn=tf.nn.tanh)

        # return tf.nn.softplus(out)

        return tf.concat(
            [
                tf.exp(tf.slice(out, [0, 0], [-1, 1])),
                tf.nn.softplus(tf.slice(out, [0, 1], [-1, 1]), name='out')
            ],
            axis=1
        )

        # return tf.concat(
        #     [
        #         tf.clip_by_value(tf.exp(tf.slice(out, [0, 0], [-1, 1])), 0, 100),
        #         tf.clip_by_value(tf.nn.softplus(tf.slice(out, [0, 1], [-1, 1])), 0, 100)
        #     ],
        #     axis=1
        # )


    def _loglikelihood(self, a_b, tte):
        location = 10.0
        growth = 20.0

        # a_b = tf.Print(a_b, [a_b])

        a = tf.slice(a_b, [0, 0], [-1, 1])
        # a = tf.exp(tf.slice(a_b, [0, 0], [-1, 1]))
        # a = tf.Print(a, [a])

        b = tf.slice(a_b, [0, 1], [-1, 1])
        # b = tf.nn.softplus(tf.slice(a_b, [0, 1], [-1, 1]))
        # b = tf.Print(b, [b])

        # tte = tf.cast(tf.reshape(tte, [-1, 1]), tf.float32)
        tte = tf.cast(tte, tf.float32)

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
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        return self.cost

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

    @staticmethod
    def _last_relevant(output, length):
        batch_size = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])

        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant


def _load_file(name):
    with open(name, 'r') as file:
        return np.loadtxt(file, delimiter=',')

def _build_data(engine, time, x, max_time, is_test):
    # y[0] will be days remaining, y[1] will be event indicator, always 1 for this data
    out_y = np.empty((0, 2), dtype=np.float32)

    # A full history of sensor readings to date for each x
    out_x = np.empty((0, max_time, 24), dtype=np.float32)

    for i in np.sort(np.unique(engine)):
        print("Engine: " + str(int(i)))
        # When did the engine fail? (Last day + 1 for train data, irrelevant for test.)
        max_engine_time = int(np.max(time[engine == i])) + 1

        if is_test:
            start = max_engine_time - 1
        else:
            start = 0

        this_x = np.empty((0, max_time, 24), dtype=np.float32)

        for j in range(start, max_engine_time):
            engine_x = x[engine == i]

            out_y = np.append(out_y, np.array((max_engine_time - j, 1), ndmin=2), axis=0)

            xtemp = np.zeros((1, max_time, 24))

            # todo: let's try padding after
            xtemp[:, 0:min(j, 99) + 1, :] = engine_x[max(0, j - max_time + 1):j + 1, :]
            # xtemp[:, max_time - min(j, 99) - 1:max_time, :] = engine_x[max(0, j - max_time + 1):j + 1, :]

            this_x = np.concatenate((this_x, xtemp))

        out_x = np.concatenate((out_x, this_x))

    return out_x, out_y

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

    X_train, y_train = _build_data(train[:, 0], train[:, 1], train[:, 2:26], max_time, False)
    X_test, y_test = _build_data(test[:, 0], test[:, 1], test[:, 2:26], max_time, False)
    joblib.dump(X_train, '../data_files/X_train.pkl')
    joblib.dump(y_train[:, 0], '../data_files/y_train.pkl')  # I don't know why he includes the column of 1s. I believe all we need is the tte.
    joblib.dump(X_test, '../data_files/X_test.pkl')
    joblib.dump(y_test[:, 0], '../data_files/y_test.pkl')  # I don't know why he includes the column of 1s. I believe all we need is the tte.


def _build_data2(engine, time, x, max_time, is_test):
    out_y = np.empty((0, 2), dtype=np.float32)

    # A full history of sensor readings to date for each x
    out_x = np.empty((0, max_time, 24), dtype=np.float32)

    for i in range(100):
        print("Engine: " + str(i))
        # When did the engine fail? (Last day + 1 for train data, irrelevant for test.)
        max_engine_time = int(np.max(time[engine == i])) + 1

        if is_test:
            start = max_engine_time - 1
        else:
            start = 0

        this_x = np.empty((0, max_time, 24), dtype=np.float32)

        for j in range(start, max_engine_time):
            engine_x = x[engine == i]

            out_y = np.append(out_y, np.array((max_engine_time - j, 1), ndmin=2), axis=0)

            xtemp = np.zeros((1, max_time, 24))
            xtemp[:, max_time-min(j, 99)-1:max_time, :] = engine_x[max(0, j-max_time+1):j+1, :]
            this_x = np.concatenate((this_x, xtemp))

        out_x = np.concatenate((out_x, this_x))

    return out_x, out_y

def data_create2():
    train = _load_file('../data_files/train.csv')
    test_x = _load_file('../data_files/test_x.csv')
    test_y = _load_file('../data_files/test_y.csv')

    all_x = np.concatenate((train[:, 2:26], test_x[:, 2:26]))
    all_x = normalize(all_x, axis=0)

    train[:, 2:26] = all_x[0:train.shape[0], :]
    test_x[:, 2:26] = all_x[train.shape[0]:, :]

    # Make engine numbers and days zero-indexed, for everybody's sanity
    train[:, 0:2] -= 1
    test_x[:, 0:2] -= 1

    # Configurable observation look-back period for each engine/day
    max_time = 100

    train_x, train_y = _build_data(train[:, 0], train[:, 1], train[:, 2:26], max_time, False)
    test_x = _build_data(test_x[:, 0], test_x[:, 1], test_x[:, 2:26], max_time, True)[0]
    # train_x, train_y = _build_data2(train[:, 0], train[:, 1], train[:, 2:26], max_time, False)
    # test_x = _build_data2(test_x[:, 0], test_x[:, 1], test_x[:, 2:26], max_time, True)[0]

    train_u = np.zeros((100, 1), dtype=np.float32)
    train_u += 1
    test_y = np.append(np.reshape(test_y, (100, 1)), train_u, axis=1)

    joblib.dump(train_x, '../data_files/X_train.pkl')
    joblib.dump(train_y, '../data_files/y_train.pkl')
    joblib.dump(test_x, '../data_files/X_test.pkl')
    joblib.dump(test_y, '../data_files/y_test.pkl')


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
    # data_create2()  # X: (20631, 100, 24), y: (20631,); X_train: (18679, 100, 24), X_test: (1952, 100, 24)
    # X_train, y_train, X_test, y_test = train_test_split(joblib.load('../data_files/X.pkl'), joblib.load('../data_files/y.pkl'), .1)
    X_train = joblib.load('../data_files/X_train.pkl')
    y_train = joblib.load('../data_files/y_train.pkl')
    X_test = joblib.load('../data_files/X_test.pkl')
    y_test = joblib.load('../data_files/y_test.pkl')

    # X_train = X_train / np.linalg.norm(X_train)
    # X_test = X_test / np.linalg.norm(X_test)

    # X_train = X_train[90:91, :, :]
    # y_train = y_train[90:91]  # 102; a = 102.5 and b = 55.2 and climbing
    # X_train = np.concatenate((X_train[90:91, :, :], X_train[190:191, :, :]))
    # y_train = np.concatenate((y_train[90:91], y_train[190:191]))
    # print(X_train); sys.exit()
    # print(y_train)  # [102. 2.]


    # todo: there must be a problem because I'm not seeing the bifurcation I'd expect in the alpha and betas across the two observations. They are essentially predicting the same death date.
    # todo: running on the data through at the same time, everything converges to the same thing. WHy?
    # todo: How do I deal with the craziness that happens when things get kind of close?

    # todo: try to get the one working in keras...and then try to figure out what the difference is between them.


    sequence_length = 100
    feature_size = 24

    data = tf.placeholder(tf.float32, [None, sequence_length, feature_size])
    target = tf.placeholder(tf.int32, [None])
    dropout = tf.placeholder(tf.float32)

    model = WtteRnn(data, target, dropout)


    batch_size = 2000
    num_epochs = 250

    sess = tf.Session()

    # tf.set_random_seed(1)
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epochs):

        random_idx = np.arange(X_train.shape[0])
        np.random.shuffle(random_idx)
        for idx in np.array_split(random_idx, X_train.shape[0] // batch_size):
            X_batch = X_train[idx]
            y_batch = y_train[idx][:, 0]
            # print(y_batch)
            # sys.exit()

        #     sess.run(model.optimize, feed_dict={data: X_batch, target: y_batch, dropout: .5})
            sess.run(model.optimize, feed_dict={data: X_batch, target: y_batch, dropout: 1})
        print(sess.run(model.prediction, feed_dict={data: X_train, target: y_train[:, 0], dropout: 1}))
        # sess.run(model.optimize, feed_dict={data: X_train, target: y_train, dropout: 1})
        # print(sess.run(model.prediction, feed_dict={data: X_train, target: y_train, dropout: 1}))

        # train_acc = sess.run(model.error, {data: X_batch, target: y_batch, dropout: 1})
        train_acc = sess.run(model.error, {data: X_train, target: y_train[:, 0], dropout: 1})
        test_acc = sess.run(model.error, {data: X_test, target: y_test[:, 0], dropout: 1})
        print('Epoch {:2d} train accuracy {:3.5f}, test accuracy {:3.5f}'.format(epoch + 1, train_acc, test_acc))


if __name__ == '__main__':
    np.set_printoptions(suppress=True, threshold=10000)
    main()


# https://github.com/gm-spacagna/deep-ttf

# todo: extend this to include censoring


# really good: https://stats.stackexchange.com/questions/352036/what-should-i-do-when-my-neural-network-doesnt-learn


# gives good guidance regarding some of the implementation nuances: https://github.com/gm-spacagna/deep-ttf/


# on masking: https://www.quora.com/What-is-masking-in-a-recurrent-neural-network-RNN


# https: // danijar.com / structuring - your - tensorflow - models /
# https://danijar.com/introduction-to-recurrent-networks-in-tensorflow/
# todo: start here: https://danijar.com/variable-sequence-lengths-in-tensorflow/


# You can use neural nets to...
# sequence classification (generating a prediction for the entire sequence)
# sequence labeling (generating a prediction for each time step)
# sequence generation (
