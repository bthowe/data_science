# Adapted from https://gist.github.com/danijar/c7ec9a30052127c7a1ad169eeb83f159
import sys
import functools
import tensorflow as tf
import matplotlib.pyplot as plt
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
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def prediction(self):
        encode1 = tf.layers.dense(self.data, 128, activation=tf.nn.relu, use_bias=True)
        encode2 = tf.layers.dense(encode1, 64, activation=tf.nn.relu, use_bias=True)
        encode3 = tf.layers.dense(encode2, 12, activation=tf.nn.relu, use_bias=True)
        encode4 = tf.layers.dense(encode3, 3, activation=tf.nn.relu, use_bias=True)

        decode1 = tf.layers.dense(encode4, 12, activation=tf.nn.relu, use_bias=True)
        decode2 = tf.layers.dense(decode1, 64, activation=tf.nn.relu, use_bias=True)
        decode3 = tf.layers.dense(decode2, 128, activation=tf.nn.relu, use_bias=True)
        decode4 = tf.layers.dense(decode3, 28 * 28, activation=tf.nn.relu, use_bias=True)

        return tf.nn.tanh(decode4)

    @lazy_property
    def cost(self):
        print(self.prediction)
        return tf.reduce_mean(tf.square(self.prediction - self.target))  # MSE

    @lazy_property
    def optimize(self):
        return tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)

    @lazy_property
    def error(self):
        return self.cost


def data_create():
    return input_data.read_data_sets('/tmp/data/', one_hot=True)


def main(mnist):
    n_epochs = 100
    batch_size = 150

    X_test = mnist.test.images

    data = tf.placeholder(tf.float32, [None, 28 * 28], name='data')
    target = tf.placeholder(tf.float32, [None, 28 * 28])
    model = SequenceClassification(data, target)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(model.optimize, {data: X_batch, target: X_batch})
        train_acc = sess.run(model.error, {data: X_batch, target: X_batch})
        test_acc = sess.run(model.error, {data: X_test, target: X_test})
        print('Epoch {:2d} train accuracy {:3.5f}, test accuracy {:3.5f}'.format(epoch + 1, train_acc, test_acc))
    tf.train.Saver().save(sess, './basic_auto_encoder')


def viz(mnist, index):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('basic_auto_encoder.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))

        graph = tf.get_default_graph()
        pred = graph.get_tensor_by_name('Tanh:0')
        data = graph.get_tensor_by_name('data:0')

        sample = mnist.test.images[index]

        plt.figure()
        plt.title('Actual')
        plt.imshow(sample.reshape((28, 28)), cmap=plt.cm.gray_r)

        image_pred = sess.run(pred, {data: sample.reshape(1, 28 * 28)})
        plt.figure()
        plt.title('Generated')
        plt.imshow(image_pred.reshape((28, 28)), cmap=plt.cm.gray_r)
        plt.show()


if __name__ == '__main__':
    data = data_create()
    # main(data)

    viz(data, 30)
