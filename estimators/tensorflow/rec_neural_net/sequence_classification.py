# Adapted from https://gist.github.com/danijar/c7ec9a30052127c7a1ad169eeb83f159
import sys
import functools
import sets
import tensorflow as tf


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class SequenceClassification:

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
        #     network = tf.contrib.rnn.GRUCell(self._num_hidden)
        #     network = tf.contrib.rnn.DropoutWrapper(network, output_keep_prob=self.dropout)
        #     return network
        #
        # network = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(self._num_layers)])
        # # network = tf.contrib.rnn.MultiRNNCell([network] * self._num_layers)  # this doesn't work! I guess it reuses the same layers.
        # output, _ = tf.nn.dynamic_rnn(network, self.data, dtype=tf.float32)
        #
        # # Select last output.
        # output = tf.transpose(output, [1, 0, 2])
        # last = tf.gather(output, int(output.get_shape()[0]) - 1)
        #
        # # Softmax layer.
        # weight, bias = self._weight_and_bias(self._num_hidden, int(self.target.get_shape()[1]))
        # prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        # return prediction
        #


        def make_cell():
            return tf.contrib.rnn.BasicRNNCell(num_units=self._num_hidden)

        network = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(self._num_layers)])
        outputs, states = tf.nn.dynamic_rnn(network, self.data, dtype=tf.float32)

        outputs_reshaped = tf.reshape(outputs, [-1, self._num_hidden * 28])
        prediction = tf.layers.dense(outputs_reshaped, 10)
        print(prediction)
        return prediction
        # return tf.nn.softmax(logits)

    @lazy_property
    def cost(self):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=self.prediction)
        loss = tf.reduce_mean(xentropy)
        return loss
        # cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
        # return cross_entropy

    @lazy_property
    def optimize(self):
        learning_rate = 0.003
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        return optimizer.minimize(self.cost)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate)
        # return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        correct = tf.nn.in_top_k(self.prediction, self.target, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        return accuracy

        # mistakes = tf.not_equal(
        #     tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        # return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)


def main():
    # # We treat images as sequences of pixel rows.
    # train, test = sets.Mnist()
    # _, rows, row_size = train.data.shape
    # num_classes = train.target.shape[1]

    rows = 28
    row_size = 28
    num_classes = 10
    n_epochs = 100
    batch_size = 150

    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets('/tmp/data/')
    X_test = mnist.test.images.reshape((-1, rows, row_size))
    y_test = mnist.test.labels

    data = tf.placeholder(tf.float32, [None, rows, row_size])
    target = tf.placeholder(tf.int32, [None])
    # target = tf.placeholder(tf.float32, [None, num_classes])
    dropout = tf.placeholder(tf.float32)
    model = SequenceClassification(data, target, dropout)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
        # for _ in range(100):
            X_batch, y_batch = mnist.train.next_batch(10)
            X_batch = X_batch.reshape((-1, rows, row_size))
            # batch = train.sample(10)
            sess.run(model.optimize, {data: X_batch, target: y_batch, dropout: 0.5})
            # sess.run(model.optimize, {data: batch.data, target: batch.target, dropout: 0.5})
        error = sess.run(model.error, {data: X_test, target: y_test, dropout: 1})
        print('Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * error))


if __name__ == '__main__':
    main()



# todo: do it with the targets one-hot encoded.
# todo: do it with the targets not one-hot encoded
# todo: include dropout
# todo: why the optimizer and archteure he uses?