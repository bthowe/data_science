import sys
import matplotlib
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt

import time
from datetime import datetime, timedelta

# import pylib.conv_widget as cw
# from pylib.tensorboardcmd import tensorboard_cmd


img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)

n_classes = 10
n_channels = 1
filt_size = [5, 5] # 5x5 pixel filters

batch_size = 50
num_iterations = 400 #1500
display_step = 100

now = datetime.now()
logs_path = now.strftime("%Y%m%d-%H%M%S") + '/summaries'
print(logs_path)

def data_create():
    return input_data.read_data_sets('/tmp/data/', one_hot=True)


def _train(data, x, y_true, loss, accuracy, optimizer, num_iterations):
    start_time = time.time()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        sess.run(init)
        train_writer = tf.summary.FileWriter(logs_path + '/train', graph=tf.get_default_graph())
        test_writer = tf.summary.FileWriter(logs_path + '/test', graph=tf.get_default_graph())

        step = 1
        for i in range(num_iterations):
            # train weights, mini-batch
            x_batch, y_true_batch = data.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: x_batch, y_true: y_true_batch})

            # Print status every 100 iterations.
            if (i % display_step == 0) or (i == num_iterations - 1):
                summary = sess.run(merged, feed_dict={x: x_batch, y_true: y_true_batch})
                train_writer.add_summary(summary, step)

                # test data on current model
                summary, l, acc = sess.run([merged, loss, accuracy], feed_dict={x: data.test.images, y_true: data.test.labels})
                test_writer.add_summary(summary, step)
                msg = "Optimization Iteration: {0:>6}, Test Loss: {1:>6}, Test Accuracy: {2:>6.1%}"
                print(msg.format(i, l, acc))

                step += 1

        time_dif = time.time() - start_time
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

        train_writer.close()
        test_writer.close()


def deep_conv_neural_net_architecture(x):
    # have to reshape the data into four dimensions (number of images X row pixels X column pixels X number of channels)
    # the -1 infers the number of rows of the flattened df (i.e., the number of distinct images), img_size is the number of pixels in one row/column of the image, and the last 1 refers to the number of channels
    x_reshape = tf.reshape(x, shape=[-1, img_size, img_size, 1])

    # send in the reshaped data and specify 32 distinct filters of dimension filt_size (5 X 5)
    out_conv = tf.layers.conv2d(
        x_reshape,
        32,
        kernel_size=filt_size,
        padding='same',
        activation=tf.nn.relu,
        name="convolution"
    )
    out_pool = tf.layers.max_pooling2d(out_conv, pool_size=(2, 2), strides=2, padding='same')
    # the output at this point will be (img_size / 2) X (img_size / 2) X filter number (32) because of the max pooling downsampling size and stride length.

    out_conv2 = tf.layers.conv2d(
        out_pool,
        64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu,
        name="convolution2"
    )
    out_pool2 = tf.layers.max_pooling2d(out_conv2, pool_size=(2, 2), strides=2, padding='same')
    # the output at this point will be (img_size / 4) X (img_size / 4) X filter number (64) because of the twice max pooled data.

    out_pool_reshape = tf.reshape(out_pool2, [-1, out_pool2.shape[1:].num_elements()])  # reshape to a (img_size / 4) * (img_size / 4) * 64 (= 3136) X 1 dimension tensor
    out = tf.layers.dense(out_pool_reshape, 1024, activation=tf.nn.relu)  # fully connected layer
    return tf.layers.dense(out, n_classes, activation=None)  # output layer (n_classes long)


def deep_conv_neural_net_run(data):
    data.test.cls = np.argmax(data.test.labels, axis=1)

    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
    y_true = tf.placeholder(tf.float32, shape=[None, n_classes], name='y_true')
    y_pred = deep_conv_neural_net_architecture(x)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    y_true_cls = tf.argmax(y_true, dimension=1)
    y_pred_cls = tf.argmax(y_pred, dimension=1)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    _train(data, x, y_true, loss, accuracy, optimizer, num_iterations)


def prediction():
    prediction = tf.argmax(y_pred, 1)

    def predict(idx):
        image = data.test.images[idx]
        return sess.run(prediction, feed_dict={x: [image]})

    idx = 0
    actual = np.argmax(data.test.labels[idx])
    print("Predicted: %d, Actual: %d" % (predict(idx), actual))
    plt.imshow(data.test.images[idx].reshape((28, 28)), cmap=plt.cm.gray_r)

if __name__ == '__main__':
    df = data_create()
    deep_conv_neural_net_run(df)

# todo: understand the train test piece in tensorboard
# todo: why not train and test like in the hierarchical nn
# todo: prediction piece
