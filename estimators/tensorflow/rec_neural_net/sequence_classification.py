# Adapted from https://gist.github.com/danijar/c7ec9a30052127c7a1ad169eeb83f159
import sys
import functools
import sets
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


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
    def __init__(self, data, target, dropout, num_classes, num_hidden=200, num_layers=3):
        self.data = data
        self.target = target
        self.dropout = dropout
        self._num_classes = num_classes
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def prediction(self):
        def make_cell():
            # return tf.contrib.rnn.GRUCell(num_units=self._num_hidden)
            return tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.BasicRNNCell(num_units=self._num_hidden),
                output_keep_prob=self.dropout
            )

        network = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(self._num_layers)])
        outputs, states = tf.nn.dynamic_rnn(network, self.data, dtype=tf.float32)

        outputs_reshaped = tf.reshape(outputs, [-1, self._num_hidden * self.data.shape[1]])  # nodes in hidden layer times the number of features which is the size of the second dimension
        return tf.layers.dense(outputs_reshaped, self._num_classes)

    @lazy_property
    def cost(self):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=self.prediction)
        return tf.reduce_mean(xentropy)

    @lazy_property
    def optimize(self):
        learning_rate = 0.001
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        correct = tf.nn.in_top_k(self.prediction, self.target, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        return accuracy


def main():
    # # We treat images as sequences of pixel rows. The pixels in a row are the features.
    rows = 28
    row_size = 28
    num_classes = 10
    n_epochs = 100
    batch_size = 150

    mnist = input_data.read_data_sets('/tmp/data/')
    X_test = mnist.test.images.reshape((-1, rows, row_size))
    y_test = mnist.test.labels

    data = tf.placeholder(tf.float32, [None, rows, row_size])
    target = tf.placeholder(tf.int32, [None])
    dropout = tf.placeholder(tf.float32)
    model = SequenceClassification(data, target, dropout, num_classes)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            X_batch = X_batch.reshape((-1, rows, row_size))
            sess.run(model.optimize, {data: X_batch, target: y_batch, dropout: 0.5})
        train_acc = sess.run(model.error, {data: X_batch, target: y_batch, dropout: 1})
        test_acc = sess.run(model.error, {data: X_test, target: y_test, dropout: 1})
        print('Epoch {:2d} train accuracy {:3.5f}, test accuracy {:3.5f}'.format(epoch + 1, train_acc, test_acc))

if __name__ == '__main__':
    main()
